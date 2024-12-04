# Intro to Formal Semantics for Natural Language

This blog deals with investigations into the meaning behind utterances in the form of a natural language. We will tackle this task more through the lens of a logician. Semantics in generative grammar that we are tackling with has been based on `‘logical’ truth-conditional semantics`. This approach relates linguistic expressions to actual states of affairs in the world by means of the concept of truth.

Semantics and Pragmatics are both concerned with ‘meaning’ and a great deal of ink has been spilt trying to define the boundaries between them. We will adopt the position that `Pragmatics = Meaning – Truth Conditions` (roughly!). For the most part we will be concerned with the meaning of sentences, rather than the meaning of utterances. 

That is, we will not be concerned with the use of sentences in actual discourse, the speech acts they can be used to perform, and so forth. From this perspective, the three sentences in (1) will all have the same meaning because they all ‘involve’ the same state of affairs.

```plaintext
    (1) a Open the window
    b The window is open
    c Is the window open

```

The fact that `a) is most likely to convey an assertion, b) a command and c) a question` is, according to this approach, a pragmatic fact about the type of speech act language users will typically associate with the _declarative, imperative and interrogative_ syntactic constructions. We will say that all the sentences of (1) convey the same proposition – the semantic ‘value’ of a sentence.

Above is just one example to have a general understanding of our mission in this blog. 

```

```

### Propositional Logic
Propositional logic, also known as sentential logic, is a branch of formal logic that deals with propositions, which are statements that are either true or false, but not both. It studies the logical relationships between propositions and how they can be combined and manipulated using logical operators.

In propositional logic, propositions are represented by variables, typically `p, q, r`, etc., and are combined using logical connectives or operators. The basic logical connectives in propositional logic include:

1. Conjunction `(AND)`: denoted by ∧ (the caret symbol), it represents the logical "and" operation. For two propositions p and q, p ∧ q is true only if both p and q are true.

2. Disjunction `(OR)`: denoted by ∨ (the wedge symbol), it represents the logical "or" operation. For two propositions p and q, p ∨ q is true if at least one of p or q is true.

3. Negation `(NOT)`: denoted by ¬ (the tilde symbol), it represents the logical "not" operation. For a proposition p, ¬p is true if p is false, and false if p is true.

4. Implication `(IF-THEN)`: denoted by → (the arrow symbol), it represents the logical "if-then" operation. For two propositions p and q, p → q is true if whenever p is true, q is also true; otherwise, it is false.

5. Biconditional `(IF AND ONLY IF)`: denoted by ↔ (the double-headed arrow symbol), it represents the logical "if and only if" operation. For two propositions p and q, p ↔ q is true if p and q have the same truth value; otherwise, it is false.

Propositional logic provides a formal framework for reasoning about `the truth or falsehood of propositions and is fundamental to various areas of computer science, mathematics, philosophy, and artificial intelligence`. 

It serves as the foundation for more complex logical systems and is widely used in areas such as theorem proving, automated reasoning, and logical circuit design.


### Predicate Calculus

Is a one step-up of propositional logic that includes `quantifiers` and `variables`. In predicate calculus, propositions are built from predicates, which are expressions that contain variables and represent properties or relations, along with constants and functions.

The key components of predicate calculus include:

1. **Predicates**: Predicates are expressions that can be true or false depending on the values of their arguments. For example, `P(x)` might represent the predicate `"x is a prime number"`, where `x` is a variable ranging over the domain of integers.

2. **Quantifiers**: Predicate calculus introduces two quantifiers, the universal quantifier `(∀)` and the existential quantifier `(∃)`. The universal quantifier `(∀)` asserts that a predicate holds for all elements in a domain, while the existential quantifier `(∃)` asserts that there exists at least one element in the domain for which the predicate holds.

3. **Variables**: Variables in predicate calculus range over a specified domain of discourse and can be used within predicates to represent unspecified objects.

4. **Functions and Constants**: Predicate calculus allows for the use of functions and constants to represent operations and specific elements within the domain of discourse.

5. **Logical Connectives**: Predicate calculus retains the logical connectives from propositional logic, such as conjunction `(∧)`, disjunction `(∨)`, negation `(¬)`, implication `(→)`, and biconditional `(↔)`, which can be used to form complex statements involving predicates.

A statement in predicate calculus can thus be more expressive and precise than in propositional logic, as it allows for quantification over elements in a domain and the formulation of statements that involve relations between objects.

### Lambda Calculus

Is another level of abstraction of natural language. Lambda calculus is a `formal system in mathematical logic and computer science` used to express computation based on function abstraction and application. It was introduced by mathematician Alonzo Church in the 1930s as part of his work on the foundations of mathematics.

In lambda calculus, computations are expressed using functions, variables, and application. The basic building blocks of lambda calculus are:

**Variables**: Variables represent values or inputs to functions.

**Abstraction**: Lambda abstraction is a way of defining anonymous functions. It allows you to define a function without explicitly naming it. The syntax for lambda abstraction is `λx. M`, where x is the parameter (variable) and M is the body of the function.

**Application**: Application is the process of applying a function to an argument. In lambda calculus, function application is denoted by placing the argument next to the function without any separator, `e.g., (λx. x + 1) 2`.

Lambda calculus is a `Turing-complete system, meaning that it can express any computable function`. It serves as the foundation for functional programming languages like `Lisp, Haskell, and Scheme`. Many concepts in computer science, such as recursion, higher-order functions, and lexical scope, can be traced back to lambda calculus.
