function [Zm_align, Z_align, perm, S] = align_sum_by_overlap(Zm, Z)


    [k1,n1] = size(Zm); [k2,n2] = size(Z);
    k = k1;

    S = Zm * Z.';      % k x k

    Sm = S;          
    perm = zeros(1,k);
    for t = 1:k
        [mx, idx] = max(Sm(:)); %#ok<ASGLU>
        [ri, cj] = ind2sub([k,k], idx);
        perm(ri) = cj;
        Sm(ri,:) = -inf;      
        Sm(:,cj) = -inf;
    end

    Zm_align = Zm;           
    Z_align  = Z(perm, :);   
end
