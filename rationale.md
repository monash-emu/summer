
# SUMMER
Scalable, Universal, Mathematical Model for Epidemic Responses

## What is SUMMER?
SUMMER is a Python-based framework for the creation and execution of compartmental (or "state-based") epidemiological models of infectious disease transmission.
It provides a range of structures for easily implementing compartmental models, including structure for some of the most commonly elaborations added to basic compartmental frameworks.

## Why SUMMER?
We saw a need for a more robust, easily extensible and universally accessible package for the construction of this sort of model.
In particular, SUMMER's aims are to:
1. Be more robust
2. Support the construction of considerably more complicated models
3. Align the process of model construction with the intuition of infectious disease epidemiology

## The existing paradigm of infectious diseases model construction
We believe that our previous workflow for infectious diseases model construction was similar to that used by modelling teams throughout the world.
Specifically, our past workflow typically followed this sequence of steps:
1. Define the modelling question
2. Create a theoretical framework for infectious disease dynamics - often consisting of a flow-diagram representing infection-related states
3. Define a system of model dynamics notated as ordinary differential equations (ODEs)
4. Convert the system of ODEs into computer code, with each compartment/model state defined separately as the sum of inflows to the compartment minus outflows
5. Run the model and analyse the outputs (including calibration, validation, etc.)

In our personal experience and from reviewing the results of others, we believe that this workflow produces considerable potential for errors.
In particular, the process of first writing a system of ODEs before converting these into model code creates an additional unnecessary step in the process with its own potential for mistakes.

## The differences of our approach
SUMMER is intended to change this paradigm, through:
* Removing the stage of writing out a system of ODEs from the workflow described above
* Encouraging modellers to think of the model constructed in terms of the epidemiological concepts that are actually being modelled, being:
  * Model compartments (i.e. infection-related states)
  * Flows (i.e. movements into, out of and between these states)
  
This approach is intended to make the modeller's intention in model construction considerably more clear and explicit.
The removal of the ODE step should reduce the potential for error and support the production of code that can directly convey the intention of the modeller to the reader.

More generally, we believe that epidemiologist and modellers think about infectious disease transmission in these terms.
That is, the intuition of an epidemic spreading through a population should and is thought of as groups of people making transitions between states.
Converting this to ODE notation moves further away from this intuition, increases the potential for errors and may create barriers to understanding for non-modellers.

## Achieving the aims

### Aim 1, more robust models
Our codebase uses standard principles of software development, such as modularity, avoidance of repetition, clear variable naming, documention, etc.
For each functional unit of code, we have written extensive tests to ensure predictable code behaviour wherever possible.

## Aim 2, more complicated models
Infectious diseases were once thought to be declining towards irrelevance, but as the current pandemic has demonstrated, continue to present major challenges. 
In particular, new technologies for control, antimicrobial resistance, newly emerging infections and other issues can make for complicated and rapidly changing policy questions.

As the modelling questions increase in complexity, it is important that modelling tools are able to keep pace and that complex models can be quickly and reliably constructed.
As the complexity of the required models increases, the risk of errors should not scale similarly.

More complicated models are also more computationally expensive.
We aim to separate the process of model construction/definition from the back-end process of obtaining the numerical solutions to the system.
This has permitted significant improvements in run-time, and creates the potential for further improvements in future versions.

## Aim 3, more intuitive models
This is described above in our critique of the previous approach to existing paradigm of infectious disease modelling above.
Ultimately, we aim for our code to be intuitive and easy to use for modellers, while potentially also being readable and transparent to non-modellers.
