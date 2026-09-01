# CMU 10-601 Spring 2023 Selected Supplements

Primary course: CMU 10-601 Introduction to Machine Learning, Spring 2023.

Official course page: <https://www.cs.cmu.edu/~mgormley/courses/10601-s23/>

Official schedule: <https://www.cs.cmu.edu/~mgormley/courses/10601-s23/schedule.html>

Official coursework page: <https://www.cs.cmu.edu/~mgormley/courses/10601-s23/coursework.html>

## Role

CMU 10-601 is not a second ML mainline.

It is used selectively to supplement Stanford CS229 with:

* alternative derivations;
* implementation-oriented viewpoints;
* experimental methodology;
* selected written/programming homework mapping;
* topics that CS229 treats more briefly.

The Stanford CS229 lecture sequence remains the repository's main course skeleton. CMU modules are attached to CS229 nodes by topic.

## Relationship to CS229

| CS229 mainline | CMU supplement | Status | Main purpose |
| --- | --- | --- | --- |
| L2-L5 | L16 MLE segment + L17 MAP/NB | current | MLE -> MAP -> NB |
| L6-L7 | L10 regularization | planned | regularization / feature practice |
| L8 | L5 model selection | planned | experimental design |
| Logistic / GLM | L8-L9 | planned | optimization / implementation |
| L13 EM/GMM | L18-L19 | later | latent-state / HMM connection |
| RL | L21-L23 | later | MDP / Q-learning |

Only selected modules are required. Unused CMU lectures are not part of the current learning plan.

## Why This Is a Supplement Layer

The existing repository uses:

* `lecture-notes/` for the Stanford CS229 lecture sequence;
* `math-derivations/` for CS229 lecture-ordered derivation records;
* `assignments/` for CS229 assignment gates and integrity boundaries;
* `syllabus-analysis/` for official CS229 resource mapping.

CMU material therefore belongs in `course-supplements/`, where it can reference CS229 nodes without becoming a parallel lecture tree. Each CMU module is named by supplement topic, not by CMU lecture number.

## Current Modules

| Module | Topic | CS229 connection | Status |
| --- | --- | --- | --- |
| [01-mle-map-naive-bayes](01-mle-map-naive-bayes/README.md) | MLE, MAP, Beta-Bernoulli, Bernoulli/Gaussian/Multinomial Naive Bayes | CS229 L4 sufficient statistics and L5 generative learning | ready for interactive study |

## Coursework Mapping

| CMU coursework | Official source | Current treatment |
| --- | --- | --- |
| HW6 - Generative Models (written) | `coursework.html` lists Homework 6 as Generative Models (written); `hw6.zip` was temporarily inspected outside the repo | parallel with CS229 PS1 / not yet started by Codex |

The temporary HW6 handout inspection found the title "Homework 6: Learning Theory and Generative Models" and sections covering CNNs, RNNs, Learning Theory, MLE/MAP, and Naive Bayes. For this supplement:

* generative subset: current;
* MLE/MAP subset: current;
* Naive Bayes subset: current;
* PAC / learning-theory subset: deferred;
* CNN/RNN parts: outside this CS229 supplement node.

No HW6 solution work is included here.

## Source Policy

Official CMU lectures and assigned readings are used as source guidance, not copied as official CMU notes. Derivations in this repository are independent reconstructions.

Tom Mitchell readings are supporting readings for the relevant supplement module. They are not copied into a third course-note track.

## Progress State

| Item | Status |
| --- | --- |
| CS229 Lecture 5 | complete |
| CS229 PS1 | in progress independently |
| CMU Supplement 01: MLE / MAP / Naive Bayes | ready for interactive study |
| CS229 Lecture 6 | not started |
