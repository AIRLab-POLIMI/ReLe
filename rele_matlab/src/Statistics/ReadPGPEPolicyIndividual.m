function individual = ReadPGPEPolicyIndividual(fp)

nbPol             = fscanf(fp, '%d', 1);
polParams         = fscanf(fp, '%f', nbPol);
nbEval            = fscanf(fp, '%d', 1);
J                 = fscanf(fp, '%f', nbEval);

nmetaparm         = fscanf(fp, '%f', 1);
difflog = zeros(nbPol, nbEval);
for i = 1:nbEval
    for j = 1:nmetaparm
        difflog(j,i) = fscanf(fp, '%f', 1);
    end
end

individual.policy  = polParams;
individual.J       = J;
individual.difflog = difflog;

end