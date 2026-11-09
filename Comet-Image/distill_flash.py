"""Distill a Comet-Image latent model into an 8-step "Flash" model.

    python distill_flash.py --teacher <run>/ema.pt --data <latents.npy> --out <runs dir> --name <name> --minutes 15

Teacher: the trained model, sampled with many small Euler steps (--teacher_steps, 32 = 4 per student step,
about the quality of 30-step sampling). Student: a copy of the teacher (same architecture) that learns to
cover each of the teacher's 4-step stretches in ONE step, so 8 student steps land where 32 teacher steps would.

Training step: take a latent, noise it to one of the 8 student times t, let the teacher run 4 small steps
from there, and train the student's velocity to jump straight to the teacher's result.
Eval: from fixed noise, how far the student's 8-step result is from the teacher's 30-step result (lower =
closer to the sharp images), next to the teacher's own 8-step result as the baseline.
"""
import argparse
import copy
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import AutoencoderKL

from dit import DiT, sample
from train_latent import VAE_ID, decode, save_grid


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--teacher", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--minutes", type=float, default=15)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--warmup", type=int, default=50)
    p.add_argument("--student_steps", type=int, default=8)
    p.add_argument("--teacher_steps", type=int, default=32)
    p.add_argument("--ema", type=float, default=0.999)
    p.add_argument("--eval_every", type=float, default=5)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    assert a.teacher_steps % a.student_steps == 0
    k = a.teacher_steps // a.student_steps

    torch.manual_seed(a.seed)
    dev = torch.device("cuda")
    out = os.path.join(a.out, a.name)
    os.makedirs(out, exist_ok=True)
    ck = torch.load(a.teacher, map_location=dev)
    c = ck["config"]
    data = torch.from_numpy(np.load(a.data)).to(dev)
    N, C, S = data.shape[0], data.shape[1], data.shape[2]
    teacher = DiT(img=S, patch=c["patch"], d=c["d"], layers=c["layers"], heads=c["heads"], in_ch=C).to(dev)
    teacher.load_state_dict(ck["state"])
    teacher.eval().requires_grad_(False)
    student = copy.deepcopy(teacher).train().requires_grad_(True)
    ema = copy.deepcopy(teacher)
    vae = AutoencoderKL.from_pretrained(VAE_ID, torch_dtype=torch.float16).to(dev).eval()
    opt = torch.optim.AdamW(student.parameters(), lr=a.lr, betas=(0.9, 0.99), weight_decay=0.0)
    print(f"[{a.name}] teacher {a.teacher} ({a.teacher_steps} steps) -> student {a.student_steps} steps, "
          f"{a.minutes} min", flush=True)

    # Fixed eval: 128 noise seeds; reference = teacher with 30 steps.
    ref30 = sample(teacher, 128, steps=30, seed=7)
    t8 = sample(teacher, 128, steps=a.student_steps, seed=7)
    base = F.mse_loss(t8, ref30).item()

    def evaluate():
        s8 = sample(ema, 128, steps=a.student_steps, seed=7)
        return round(F.mse_loss(s8, ref30).item(), 4)

    print(json.dumps({"min": 0.0, "teacher_8step_vs_30step": round(base, 4), "student_8step_vs_30step": evaluate()}),
          flush=True)
    save_grid(decode(vae, sample(teacher, 16, steps=a.student_steps, seed=1)), os.path.join(out, "grid_teacher_8steps.png"))
    save_grid(decode(vae, sample(teacher, 16, steps=30, seed=1)), os.path.join(out, "grid_teacher_30steps.png"))

    def save_ckpt(step, t_train):
        torch.save({"config": c | {"distilled_from": a.teacher}, "img": S, "in_ch": C, "step": step,
                    "train_min": round(t_train / 60, 2), "steps": a.student_steps, "state": ema.state_dict()},
                   os.path.join(out, "ema.pt"))

    budget, t_train, step, seen, losses = a.minutes * 60, 0.0, 0, 0, []
    next_log, next_eval = 60.0, a.eval_every * 60
    dt = 1.0 / a.teacher_steps
    while t_train < budget:
        t0 = time.perf_counter()
        frac = t_train / budget
        lr = a.lr * min(1.0, step / max(1, a.warmup)) * (1.0 if frac < 0.8 else 1 - (frac - 0.8) / 0.2 * 0.9)
        for g in opt.param_groups:
            g["lr"] = lr
        x1 = data[torch.randint(0, N, (a.batch,), device=dev)].float()
        flip = torch.rand(a.batch, device=dev) < 0.5
        x1 = torch.where(flip[:, None, None, None], x1.flip(3), x1)
        x0 = torch.randn_like(x1)
        t = torch.randint(0, a.student_steps, (a.batch,), device=dev).float() / a.student_steps
        xt = (1 - t)[:, None, None, None] * x0 + t[:, None, None, None] * x1
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            y = xt
            for j in range(k):
                y = y + teacher(y, t + j * dt).float() * dt
            target = (y - xt) * a.student_steps  # velocity that jumps there in one student step
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = F.mse_loss(student(xt, t).float(), target)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        opt.step()
        with torch.no_grad():
            for pe, pm in zip(ema.parameters(), student.parameters()):
                pe.lerp_(pm, 1 - a.ema)
        torch.cuda.synchronize()
        t_train += time.perf_counter() - t0
        step += 1
        seen += a.batch
        losses.append(loss.item())
        if t_train >= next_log:
            print(json.dumps({"min": round(t_train / 60, 2), "step": step, "loss": round(float(np.mean(losses)), 4),
                              "img_per_sec": round(seen / t_train), "vram_mb": round(torch.cuda.max_memory_allocated() / 2**20),
                              "train_min_left": round((budget - t_train) / 60, 1)}), flush=True)
            losses, next_log = [], next_log + 60
        if os.path.exists(os.path.join(out, "STOP")):
            print(f"STOP file found at {t_train / 60:.1f} min", flush=True)
            break
        if t_train >= next_eval:
            save_ckpt(step, t_train)
            save_grid(decode(vae, sample(ema, 16, steps=a.student_steps, seed=1)),
                      os.path.join(out, f"grid_{t_train / 60:04.1f}min_student_{a.student_steps}steps.png"))
            print(f"EVAL {t_train / 60:.1f} min", json.dumps({"student_8step_vs_30step": evaluate()}), flush=True)
            next_eval += a.eval_every * 60

    save_ckpt(step, t_train)
    save_grid(decode(vae, sample(ema, 16, steps=a.student_steps, seed=1)),
              os.path.join(out, f"grid_final_student_{a.student_steps}steps.png"))
    summary = {"name": a.name, "steps": step, "train_minutes": round(t_train / 60, 2), "img_per_sec": round(seen / t_train),
               "teacher_8step_vs_30step": round(base, 4), "student_8step_vs_30step": evaluate()}
    json.dump(summary, open(os.path.join(out, "summary.json"), "w"), indent=2)
    print("DONE", json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
