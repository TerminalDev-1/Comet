"""Generate story-reasoning examples with visible thinking:

    <story>
    Question: <question>
    <think><step>. <step>. So <conclusion>.</think>
    Answer: <answer>

    python data/gen_think.py            # writes data/think_{train,val}.bin + data/think_test.jsonl

Train/val use one set of names and objects; the test set uses NEW names and objects the model
never saw in this format, so a right answer means it reasons instead of copying.
The bins mix the reasoning examples 50/50 with TinyStories, so it keeps its English.
"""
import json
import os
import random

import numpy as np
from tokenizers import Tokenizer

HERE = os.path.dirname(os.path.abspath(__file__))

REAL_NAMES = """Tom Sue Lily Ben Mia Max Anna Sam Tim Lucy Jack Emma Leo Zoe Mark Amy Jane Bob Kate Sara Adam Alice Andy
Bella Billy Carl Chloe Daisy David Eddie Ella Emily Eric Eva Fred Grace Greg Hannah Harry Henry Holly Jenny Jill Joe
John Josh Julia Kevin Kim Kyle Laura Liam Lisa Luke Mary Matt Megan Mike Nina Noah Nora Oscar Paul Peter Polly Rachel
Ray Rita Rob Rose Ryan Sally Sophie Steve Susan Ted Tina Toby Tony Victor Will Wendy Abby Alex Annie Beth Brian Carla
Charlie Cindy Danny Dora Ellie Felix Fiona Frank Gina George Hugo Iris Jason Jess Judy Karen Kelly Lara Leah Lena
Lewis Lola Lou Maria Martin Nancy Ned Nick Olivia Owen Pam Pete Ricky Sean Simon Stella Ted Tara Tess Vera Wally Zack
Amber Arthur Betty Clara Colin Dean Elsa Gary Gus Hazel Ian Jade Joy June Kai Lucas Maya Mila Nate Neil Otto Penny""".split()
SYL = ["ba", "bi", "bo", "da", "di", "do", "fa", "fi", "ga", "go", "ka", "ki", "ko", "la", "li", "lo", "ma", "mi",
       "mo", "na", "ni", "no", "pa", "pi", "po", "ra", "ri", "ro", "sa", "si", "so", "ta", "ti", "to", "va", "vi", "za",
       "zo", "bel", "dan", "fen", "gil", "jon", "kel", "lin", "mar", "nel", "pim", "ral", "sim", "tam", "vin", "wil"]
TEST_NAMES = ["Ollie", "Ruby", "Finn", "Ivy", "Rosie", "Jake", "Molly", "Dan"]


def made_up_name(r):
    n = "".join(r.choice(SYL) for _ in range(r.choice([2, 2, 3])))
    return n.capitalize()


class Names(list):
    """Real names plus a fresh made-up name about half the time, so memorizing names can't work."""

    def __init__(self, real, r):
        super().__init__(real)
        self.r = r

    def pick(self, k):
        out = []
        while len(out) < k:
            n = made_up_name(self.r) if self.r.random() < 0.5 else self.r.choice(self)
            if n not in out and n not in TEST_NAMES:
                out.append(n)
        return out


TRAIN_NAMES = [n for n in dict.fromkeys(REAL_NAMES) if n not in TEST_NAMES]
TEST_NAMES = ["Ollie", "Ruby", "Finn", "Ivy", "Rosie", "Jake", "Molly", "Dan"]
TRAIN_THINGS = """ball doll apple book kite hat cookie cup car truck block crown flower ring key coin stick stone
leaf feather boat train drum bell box_of_crayons pencil rope bone carrot cake pie candy toy_robot puzzle scarf glove
shoe button marble wand card letter cap blanket pillow plate bowl jar brush comb mirror lamp clock map flag star
cherry pear orange lemon egg""".replace("_", " ").split()
TEST_THINGS = ["shell", "crayon", "banana", "spoon", "balloon", "sock"]
TRAIN_PLACES = """box bag basket drawer car tree closet cupboard sink bath barn shed garden kitchen bed chair nest
cave boat truck house hole pond yard attic shelf oven fridge wagon tub""".split()
TEST_PLACES = ["bucket", "pocket", "tent"]
ANIMALS = [("dog", "barked", "bark"), ("cat", "meowed", "meow"), ("bird", "sang", "sing"), ("fish", "swam", "swim"),
           ("duck", "quacked", "quack"), ("cow", "mooed", "moo")]
PEOPLE = ["girl", "boy"]
NUM = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"]
FEEL = [("lost {p} {t}", "Losing something you love makes you sad", "sad"),
        ("got a new {t} as a present", "Getting a present makes you happy", "happy"),
        ("heard a loud noise in the dark", "Loud noises in the dark make you scared", "scared"),
        ("broke {p} {t} by accident", "Breaking something you like makes you sad", "sad"),
        ("won the race at school", "Winning makes you proud", "proud"),
        ("found {p} lost {t}", "Finding something you lost makes you happy", "happy")]


def pro(name):
    return "her" if name in {"Sue", "Lily", "Mia", "Anna", "Lucy", "Emma", "Zoe", "Amy", "Jane", "Kate",
                             "Sara", "Ruby", "Ivy", "Rosie", "Molly"} else "his"


def ex_give(r, N, T, P):
    a, b = N.pick(2)
    t = r.choice(T)
    story = f"{a} had a {t}. {a} gave the {t} to {b}."
    think = f"{a} had the {t}. {a} gave the {t} to {b}. So {b} has the {t} now."
    return story, f"Who has the {t}?", think, b


def ex_give2(r, N, T, P):
    a, b, c = N.pick(3)
    t = r.choice(T)
    story = f"{a} had a {t}. {a} gave the {t} to {b}. Then {b} gave the {t} to {c}."
    think = f"{a} had the {t}. {a} gave it to {b}. Then {b} gave it to {c}. So {c} has the {t} now."
    return story, f"Who has the {t}?", think, c


def ex_place(r, N, T, P):
    a = N.pick(1)[0]
    t1, t2 = r.sample(T, 2)
    p1, p2 = r.sample(P, 2)
    story = f"{a} put the {t1} in the {p1} and the {t2} in the {p2}."
    q, t, p = (t1, t1, p1) if r.random() < 0.5 else (t2, t2, p2)
    think = f"{a} put the {t1} in the {p1}. {a} put the {t2} in the {p2}. So the {t} is in the {p}."
    return story, f"Where is the {q}?", think, f"in the {p}"


def ex_move(r, N, T, P):
    a, b = N.pick(2)
    t = r.choice(T)
    p1, p2 = r.sample(P, 2)
    story = f"{a} put the {t} in the {p1}. Later, {b} moved the {t} to the {p2}."
    think = f"First the {t} was in the {p1}. Then {b} moved it to the {p2}. So the {t} is in the {p2} now."
    return story, f"Where is the {t}?", think, f"in the {p2}"


def ex_count(r, N, T, P):
    a = N.pick(1)[0]
    t = r.choice(T)
    n = r.randint(2, 9)
    if r.random() < 0.5:
        m = r.randint(1, n - 1)
        story = f"{a} had {NUM[n]} {t}s. {a} gave {NUM[m]} {t}s to a friend."
        think = f"{a} had {NUM[n]} {t}s. {a} gave away {NUM[m]}. {NUM[n].capitalize()} take away {NUM[m]} is {NUM[n - m]}."
        k = n - m
    else:
        m = r.randint(1, 10 - n)
        story = f"{a} had {NUM[n]} {t}s. Mom gave {a} {NUM[m]} more {t}s."
        think = f"{a} had {NUM[n]} {t}s. Mom gave {NUM[m]} more. {NUM[n].capitalize()} and {NUM[m]} make {NUM[n + m]}."
        k = n + m
    return story, f"How many {t}s does {a} have now?", think, NUM[k]


def ex_animal(r, N, T, P):
    a, b = N.pick(2)
    an, verb, can = r.choice(ANIMALS)
    pe = r.choice(PEOPLE)
    story = f"{a} is a {an} and {b} is a {pe}. One of them {verb}." if r.random() < 0.5 else         f"{b} is a {pe} and {a} is a {an}. One of them {verb}."
    think = f"A {an} can {can}. A {pe} cannot {can}. {a} is the {an}. So {a} {verb}."
    return story, f"Who {verb}?", think, a


def ex_compare(r, N, T, P):
    a, b = N.pick(2)
    big, small = r.choice([("taller", "shorter"), ("bigger", "smaller"), ("older", "younger"), ("faster", "slower")])
    story = f"{a} is {big} than {b}."
    if r.random() < 0.5:
        think = f"{a} is {big} than {b}. So {b} is {small} than {a}."
        return story, f"Who is {small}?", think, b
    think = f"{a} is {big} than {b}. So {a} is the {big} one."
    return story, f"Who is {big}?", think, a


def ex_feel(r, N, T, P):
    a = N.pick(1)[0]
    t = r.choice(T)
    ev, rule, f = r.choice(FEEL)
    story = f"{a} {ev.format(p=pro(a), t=t)}."
    think = f"{rule}. {a} {ev.format(p=pro(a), t=t)}. So {a} felt {f}."
    return story, f"How did {a} feel?", think, f


KINDS = [ex_give, ex_give2, ex_place, ex_move, ex_count, ex_animal, ex_compare, ex_feel]


def render(story, q, think, ans, with_answer=True):
    head = f"{story}\nQuestion: {q}\n<think>"
    return head + (f"{think}</think>\nAnswer: {ans}" if with_answer else "")


def make(n, seed, names, things, places):
    r = random.Random(seed)
    names = Names(names, r) if names is not TEST_NAMES else Names(names, r)
    if names[0] in TEST_NAMES:  # test set: only the held-out real names, no made-up ones
        names.pick = lambda k: r.sample(TEST_NAMES, k)
    out = []
    for i in range(n):
        kind = KINDS[i % len(KINDS)]
        story, q, think, ans = kind(r, names, things, places)
        out.append((kind.__name__[3:], story, q, think, ans))
    return out


def main():
    tok = Tokenizer.from_file(os.path.join(HERE, "tokenizer.json"))
    eot = tok.token_to_id("<|endoftext|>")
    ts = np.memmap(os.path.join(HERE, "train.bin"), dtype=np.uint16, mode="r")
    for split, n, seed in (("train", 120_000, 1), ("val", 2_000, 2)):
        ids = []
        for _, s, q, t, a in make(n, seed, TRAIN_NAMES, TRAIN_THINGS, TRAIN_PLACES):
            ids += tok.encode(render(s, q, t, a)).ids + [eot]
        k = len(ids)
        start = 0 if split == "train" else 50_000_000
        mixed = np.concatenate([np.array(ids, dtype=np.uint16), np.asarray(ts[start:start + k])])
        mixed.tofile(os.path.join(HERE, f"think_{split}.bin"))
        print(f"think_{split}.bin: {k:,} reasoning tokens + {k:,} TinyStories tokens")
    with open(os.path.join(HERE, "think_test.jsonl"), "w") as f:
        for kind, s, q, t, a in make(400, 3, TEST_NAMES, TEST_THINGS, TEST_PLACES):
            f.write(json.dumps({"kind": kind, "prompt": render(s, q, t, a, with_answer=False), "answer": a,
                                "think": t}) + "\n")
    print("think_test.jsonl: 400 held-out examples (new names and objects)")
    print("\nExample:\n" + render(*make(1, 9, TRAIN_NAMES, TRAIN_THINGS, TRAIN_PLACES)[0][1:]))


if __name__ == "__main__":
    main()
