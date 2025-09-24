function [F] = updateF1(H,F_old,YY,W)
%H：n*m
%F_old: n*c
%YY : n*c
%W:1*m
dv=1./sqrt(max(sum(bsxfun(@times,H,W),2),eps));
de=1./max(sum(H),eps);
v1 = bsxfun(@times, dv, H); % O(nc)
v3 = bsxfun(@times, W, de); % O(size of (W))

for tt = 1:10
    v2 = v1' * F_old; % O(nc^2)
    v4 = diag(v3) * v2; % < O(n)
    v5 = v1 * v4; % O(nc^2)
    v6 = v5 + YY;
    [U1, ~, V1] = svd(v6, 'econ'); % O(nc^2)
    F = U1 * V1';
    diff = abs(norm(F - F_old,'fro')/norm(F_old,'fro'));
    if diff<1e-2
        break
    else
    F_old = F;
    end
end

end
