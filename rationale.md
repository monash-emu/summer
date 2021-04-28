# SUMMER
Scalable, Universal, Mathematical Model for Epidemic Responses

## What is SUMMER?
SUMMER is a Python-based framework for the creation and execution of compartmental (or "state-based") epidemiological models of infectious diseases transmission.
It provides a range of structures for easily implementing compartmental models, including structure for some of the most commonly elaborations added to basic compartmental frameworks.

## Why SUMMER?
We saw a need for a more robust, easily extensible and universally accessible package for the construction of this sort of model.
In particular, SUMMER aims to:
* Be more robust
* Support the construction of considerably more complicated models

## The existing paradigm of infectious diseases model construction
We believe that our previous workflow for  infectious diseases model construction was similar to that used by modelling teams throughout the world.
Specifically, our workflow generally followed this sequence of steps:
1. Define the modelling question
2. Create a theoretical framework for infectious diseases dynamics - usually consisting of a flow-diagram representing infection-related states
3. Define a system of model dynamics notated as ordinary differential equations (ODEs)
4. Convert the system of ODEs into computer code, with each compartment/model state defined separately according to its inflows and outflows
5. Run the model and analyse the outputs (including calibration, validation, etc.)
In our personal experience and from reviewing the results of others, we believe that this workflow produces considerable potential for errors.
In particular, the process of first writing a system of ODEs before converting these into model code creates an additional unnecessary step in the process with its own potential for mistakes.

## Our proposed paradigm
SUMMER is intended to change this paradigm, through:
* Removing the stage of writing out a system of ODEs from the workflow described above
* Encouraging modellers to think of the model constructed in terms of the epidemiological concepts that are actually being modelled, being:
  * Model compartments (i.e. infection-related states)
  * Flows (i.e. movements into, out of and between these states)
This approach is intended to make it considerably more clear and explicit as to what the modeller is intending through their model construction.
The removal of the ODE step should reduce the potential for error and support the production of code that can directly convey the intention of the modeller to the reader.
