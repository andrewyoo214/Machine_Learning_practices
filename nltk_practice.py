"""
Natural Language Toolkit Practice
"""

from nltk.inference.resolution import *
from nltk.sem import logic
from nltk.sem.logic import *
read_expr = logic.Expression.fromstring

p1 = read_expr('all x.(man(x) -> mortal(x))')
p2 = read_expr('man(Socrates)')
c=read_expr('mortal(Socrates)')
logic._counter._value = 0

tp = ResolutionProverCommand(c, [p1,p2])
tp.prove()
print(tp.proof())

r1 = read_expr('all x.(King(x) & Greedy(x) -> Evil(x))')
r2 = read_expr('King(John)')
r3 = read_expr('all y.(Greedy(y))')
r4 = read_expr('Brother(Richard, John)')
c = read_expr('Evil(John)')
tp = ResolutionProverCommand(c, [r1, r2, r3, r4])
print(tp.prove())
print(tp.proof())

r1 = read_expr('all x.(King(x) & Greedy(x) -> Evil(x))')
r2 = read_expr('King(John)')
r3 = read_expr('all y.(Greedy(y))')
r4 = read_expr('Brother(Richard, John)')
c = read_expr('-Evil(John)')
tp = ResolutionProverCommand(c, [r1, r2, r3, r4])
print(tp.prove())
print(tp.proof())


##let's try with find_answer

r1 = read_expr('all x.(King(x) & Greedy(x) -> Evil(x))')
r2 = read_expr('King(John)')
r3 = read_expr('all y.(Greedy(y))')
r4 = read_expr('Brother(Richard, John)')
#c = read_expr('Evil(John)')
c = read_expr('all x.(Evil(x) -> ANSWER(x))')
tp = ResolutionProverCommand(None, [c, r1, r2, r3, r4]) #증명하려는 결론이 아니라 evil인 애가 누군지 answer를 찾으려 하는 것.
sorted(tp.find_answers())
#print(tp.prove())
print(tp.proof())


"""
"The Law says that it is a crime for an American to sell weapons to hostile nations. The country Nono, and enemy of America, has some missiles, and all of its missiles were sold to it by Colonel West, who is American"
"""

from nltk.inference.resolution import *
from nltk.sem import logic
from nltk.sem.logic import *
read_expr = logic.Expression.fromstring
logic._counter._value = 0

r1 = read_expr('American(x) & Weapon(y) & Sells(x, y, z) & Hostile(z) -> Criminal(x)')
r2 = read_expr('Owns(Nono, M1)')
r3 = read_expr('Missile(M1)')
r4 = read_expr('Missile(x) & Owns(Nono, x) -> Sells(West, x, Nono)')
r5 = read_expr('Missile(x) -> Weapon(x)')
r6 = read_expr('Enemy(x, America) -> Hostile(x)')
r7 = read_expr('American(West)')
r8 = read_expr('Enemy(Nono, America)')
c = read_expr('Criminal(West)')


tp = ResolutionProverCommand(c, [r1, r2, r3, r4, r5, r6, r7, r8])

print(tp.prove())
print(tp.proof())
