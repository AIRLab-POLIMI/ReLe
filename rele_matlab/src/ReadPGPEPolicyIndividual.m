function individual = ReadPGPEPolicyIndividual(fp)

nbPol             = fscanf(fp, '%d');
polParams         = fscanf(fp, '%f', nbPol);
nbEval            = fscanf(fp, '%d');
J                 = fscanf(fp, '%f', nbEval);

difflog = zeros(nbPol, nbEval);
for i = 1:nbPol
    for j = 1:nbEval
        difflog(i,j) = fscanf(fp, '%f');
    end
end

individual.policy  = polParams;
individual.J       = J;
individual.difflog = polParams;

end