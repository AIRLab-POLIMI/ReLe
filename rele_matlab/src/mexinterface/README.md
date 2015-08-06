
MEX interface
--------------------

This folder provides the access to ReLe functions via MATLAB.

First you need to compile the mex functions through MATLAB.
Remember to check the last supported version of GCC. For example, MATLAB R2015a supports up to GCC 4.7.x
You need to move into *mexinterface/* where you can use the script *MEXMakefile* to compile the functions.
Then add such folder to the MATLAB search path.

Provided functions
--------------------

# collectSamples
~~~~
[new_samples, ret, G, H] = collectSamples(domain_settings, nbEpisodes, maxSteps, gamma, [params])
~~~~
To use *collectSamples* you need to define the domain settings. The domain settings defines all the elements required to simulate the environment.
In particular the following elements are defined in the settings:
1. domain that must be simulated with eventually all the associated parameters
2. policy
Finally, a last parameters can be given to the function. This input, named *params*, is usually a structure that contains additional parameters required by the given domain settings.

### Inputs:
- name of the domain settings
- number of episodes
- maximum number of steps
- optional additional parameters

### Outputs:
- trajectories
- expected return per trajectory
- gradient
- hessian

### All the admissible calls are reported here:
[new_samples, ret, G, H] = collectSamples(domain_settings, nbEpisodes, maxSteps, gamma)
[new_samples, ret, G, H] = collectSamples(domain_settings, nbEpisodes, maxSteps, gamma, params)


### How to define a new settings
You have to perform these simple steps:
1. Define a function that represents the new settings *NEWSET_domain_settings* in *CSDomainSettings.h* with inputs 
~~~~
void
NEWSET_domain_settings(
    int nlhs, mxArray *plhs[], /* Output variables */
    int nrhs, const mxArray *prhs[] /* Input variables */
);
~~~~
2. Update the function *CollectSamplesGateway* in CSDomainSettings.h to handle the new settings. We suggest to use the prefix *NEWSET* as the name of the settings.

MDPs can be grouped based on the representation of state and action spaces (continuous or finite). This difference will influence the definition of the function *NEWSET_domain_settings*. We provide a general structure of such function but you can refer to the implementations in *CSDomainSettings.cpp* for more details.

~~~~
void
NEWSET_domain_settings(
    int nlhs, mxArray *plhs[], /* Output variables */
    int nrhs, const mxArray *prhs[] /* Input variables */
)
{
    GETSETTINGS; // it is always the first instruction, it defines several variables (nbEpisodes, nbSteps, gamma)
    
    // extract information from the struct containing the parameter (if any)

    // define domain
    
    // define policy
    
    //get dataset (defines first and second outputs: DS and J)
    //continuous domain
    //SAMPLES_GATHERING(DenseAction, DenseState, mdp.getSettings().continuosActionDim, mdp.getSettings().continuosStateDim)
    //discrete domain
    //SAMPLES_GATHERING(FiniteAction, DenseState, 1, mdp.getSettings().continuosStateDim)
    
    //return additional value starting from third

}
~~~~
