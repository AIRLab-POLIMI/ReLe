function individual = ReadPGPEPolicyIndividual(fp)

nbPol             = fscanf(fp, '%d', 1);
polParams         = fscanf(fp, '%f', nbPol);
nbEval            = fscanf(fp, '%d', 1);
J                 = fscanf(fp, '%f', nbEval);

difflog = zeros(nbPol, nbEval);
for i = 1:nbPol
    for j = 1:nbEval
        difflog(i,j) = fscanf(fp, '%f', 1);
    end
end

individual.policy  = polParams;
individual.J       = J;
individual.difflog = difflog;

end