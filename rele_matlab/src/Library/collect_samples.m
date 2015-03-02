function [new_samples, uJ, dJ, entropy] = collect_samples(domain, ...
    maxepisodes, maxsteps, policy, avg_rew_setting, gamma)

%%% Initialize some variables
simulator = [domain '_simulator'];
initialize_state = [domain '_initialize_state'];

%%% Initialize simulator
feval(simulator);

empty_sample.s = [];
empty_sample.a = [];
empty_sample.r = [];
empty_sample.nexts = [];
empty_sample.terminal = [];
new_samples = repmat(empty_sample,1, maxepisodes);

uJ = 0;
dJ = 0;
entropy = 0;

%%% Main loop
parfor episodes = 1 : maxepisodes
    
    new_samples(episodes).policy = policy;
    
    %%% Select initial state
    initial_state = feval(initialize_state, simulator);
    
    %%% Run one episode (up to the max number of steps)
    [epi_samples, totdrew, toturew, He] = ...
        execute(domain, initial_state, simulator, policy, maxsteps, ...
        avg_rew_setting, gamma);
    dJ = dJ + totdrew;
    uJ = uJ + toturew;
    entropy = entropy + He;
    
    %%% Store the new samples
    if ~isempty(epi_samples.a)
        new_samples(episodes).s = epi_samples.s;
        new_samples(episodes).a = epi_samples.a;
        new_samples(episodes).r = epi_samples.r;
        new_samples(episodes).nexts = epi_samples.nexts;
        new_samples(episodes).terminal = epi_samples.terminal;
    end
    
end

uJ = uJ / maxepisodes;
dJ = dJ / maxepisodes;
entropy = entropy / maxepisodes;

return
