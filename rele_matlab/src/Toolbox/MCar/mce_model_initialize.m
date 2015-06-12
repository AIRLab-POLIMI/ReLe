function model = mce_model_initialize(xLimit, vLimit)
%==================================================================================================
% Inizializza il modello della mountain car
%
% ___ SYNTAX __________________________________________________________________
% @NAME
% model = room_model_initialize()
%
%==================================================================================================

model.dt = 0.1;  % Passo d'integrazione
model.mass = 1;   % massa
model.g = 9.81;   % accellerazione di gravita'

% Limiti intervallo della posizione
model.xLB = xLimit(1);
model.xUB = xLimit(2);

% Limiti intervallo della velocit??
model.vLB = vLimit(1);
model.vUB = vLimit(2);

% Azioni disponibili
model.nU  = 3;
model.throttle = [-4 0 4]';

end
