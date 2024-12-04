# Finite State Machine and Automata

A finite automaton is the most basic machine capable of pattern recognition (FA). It is used to describe a Regular Language, just like in `/baa+!/`.

It is also used to recognize and evaluate Natural Language Expressions. There are five parts, or tuples, to the abstract machine called the finite automata or finite state machine. Its various states and the rules for transitioning between them are contingent upon the input symbol that is utilized.

Depending on the states and the regulations, the input string may be accepted or denied. It is essentially a model of an abstract digital computer that interprets an input string and adjusts its internal state based on the input symbol that is being received at that moment.

The following characteristics of automata in the aforementioned figure are

---

![finite_fig1](../extras/finite_automata/fig1.png)

---

A Finite Automata consists of the following:

```
Q : Finite set of states.       
Σ : set of Input Symbols.       
q : Initial state.       
F : set of Final States.       
δ : Transition Function.

```

Formal specification of machine is

```
{ Q, Σ, q, F, δ }

```

1. Deterministic Finite Automata (DFA):

```python
DFA consists of 5 tuples {Q, Σ, q, F, δ}.
Q : set of all states.
Σ : set of input symbols. ( Symbols which machine takes as input )
q : Initial state. ( Starting state of a machine )
F : set of final state.
δ : Transition Function, defined as δ : Q X Σ --> Q.

```

In a DFA, the machine only enters one state for a specific input character. For every input symbol, a transition function is defined for each state. Additionally, DFA does not support null (or) moves, meaning that it cannot change its state without an input character.

```python
Create a DFA that, for instance, only accepts strings that end in 'a'.

    Assumed: Σ = {a,b}, q = {q0}, F={q1}, Q = {q0, q1}

If you want to create a state transition diagram that is accurate, you should first take into account a language set of all the potential acceptable strings.

    L = {a, a, a, a, a, aa, aaa, aaaa, ba, bba, bbbaa, aba, abba, aaba, abaa}

The list above is just a small portion of all possible acceptable strings; there are many other strings that contain the letters "a" and "b".
```

---

![finite_fig2](../extras/finite_automata/fig2.png)

---



| State / Symbol | a             | b       |
| :------------- | :------------ | :------ |
| $q_0$          | ${q_1}$       | $q_0$   |
| $q_1$          | ${q_1}$       | $q_0$   |




2) Nondeterministic Finite Automata(NFA): NFA is similar to DFA except following additional features:

1. Null (or ?) move is allowed i.e., it can move forward without reading symbols.
2. Ability to transmit to any number of states for a particular input.




See below figure:
    As you can see in the transition function is for any input including null (or ?), NFA can go to any state number of states. For example, below is an NFA for the above problem.

| State / Symbol | a             | b       |
| :------------- | :------------ | :------ |
| $q_0$          | ${q_0,q_1}$   | $q_0$   |
| $q_1$          | ?             | ?       |

---

![finite_fig2](../extras/finite_automata/fig3.png)

---

One important thing to note is, in NFA, if any path for an input string leads to a final state, then the input string is accepted.

Since all the tuples in DFA and NFA are the same except for one of the tuples, which is Transition Function (?)

```
In case of DFA
? : Q X ? --> Q
In case of NFA
? : Q X ? --> $2^Q$  
```

ℹ️ Now if you observe you’ll find out `Q X ? –> Q` is part of `Q X ? –> 2 ^ Q`.

On the RHS side, Q is the subset of $2^Q$ which indicates Q is contained in $2^Q$ or Q is a part of $2^Q$, however, the reverse isn’t true. So mathematically, we can conclude that every DFA is NFA but not vice-versa. Yet there is a way to convert an NFA to DFA, so there exists an equivalent DFA for every NFA.

- Both NFA and DFA have the same power and each NFA can be translated into a DFA.
- There can be multiple final states in both DFA and NFA.
- NFA is more of a theoretical concept.
- DFA is used in Lexical Analysis in Compiler.
- If the number of states in the NFA is N then, its DFA can have maximum $2^N$ number of states.


# Onto the Discussion about Determinism

The concept of deterministic versus non-deterministic automata is fundamental in computer science, particularly in the theory of computation and formal language theory. Here are some areas where this concept is applied:

1. **Finite Automata**: Deterministic Finite Automata (DFA) and Non-deterministic Finite Automata (NFA) are used to recognize regular languages. DFAs have a single unique transition from each state for each input symbol, while NFAs can have multiple transitions for the same input symbol from a single state.

2. **Regular Expressions**: Regular expressions are often used to describe patterns in strings. There are algorithms to convert between regular expressions and NFAs, and between NFAs and DFAs.

3. **Compiler Design**: Lexical analysis, which is the first phase of compilation, often involves converting regular expressions describing the language's tokens into NFAs or DFAs for efficient processing.

4. **Network Protocol Specification**: Automata theory is used to model and verify network protocols. Deterministic and non-deterministic finite automata are used to represent the behavior of the protocols.

5. **Model Checking**: Model checking is a technique used to verify whether a system meets a given specification. Automata are used to model the system's behavior, and determining whether it's deterministic or non-deterministic can affect the complexity of the verification process.

6. **Software Verification**: Automata are used in software verification techniques such as symbolic execution and abstract interpretation to analyze and verify the behavior of software systems.

7. **Natural Language Processing**: Automata theory can be applied to tasks like tokenization and morphological analysis in natural language processing.

8. **Database Query Optimization**: Finite automata are used in database query optimization, particularly in pattern matching and indexing.

These are just a few examples of where the concepts of deterministic versus non-deterministic automata find applications. They form a foundational part of theoretical computer science and are applied in various fields where formal language and computational models are relevant.


# What Does Being A Deterministic vs Non-deterministic Machine Entail?

In the context of automata theory and computational theory, "deterministic" and "non-deterministic" refer to two different types of machines: deterministic machines and non-deterministic machines. These terms describe how these machines make decisions and transitions between states.

1. **Deterministic**: A deterministic machine is one where, given a particular state and input symbol, there is only one possible next state. In other words, the behavior of the machine is entirely determined by its current state and the input it receives. There are no ambiguities or choices in the transitions between states. Deterministic Finite Automata (DFA) are examples of deterministic machines.

2. **Non-deterministic**: A non-deterministic machine is one where, given a particular state and input symbol, there can be multiple possible next states. This means that the machine can "guess" which state to transition to based on the current state and input. Non-deterministic Finite Automata (NFA) are examples of non-deterministic machines.

Here are some key differences between deterministic and non-deterministic machines:

- **Transitions**: Deterministic machines have exactly one possible transition for each state and input symbol pair, while non-deterministic machines may have multiple possible transitions.

- **Decision Making**: Deterministic machines make decisions based solely on their current state and the input symbol they receive, whereas non-deterministic machines can make "guesses" about which state to transition to.

- **Ambiguity**: Deterministic machines have no ambiguity in their transitions, whereas non-deterministic machines may have ambiguity due to multiple possible transitions.

In summary, deterministic machines follow a single, predictable path of execution based on their current state and input, while non-deterministic machines have the ability to explore multiple paths simultaneously or make non-deterministic choices during computation.
