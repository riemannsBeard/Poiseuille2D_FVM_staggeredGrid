function grid = stretching(alpha, N)

%     grid = 0.5*(1 - tanh(alpha*(1 - 2*(0:1:N)/N))./tanh(alpha));

    grid = (0:1:N)/N - (alpha/(2*pi))*sin(2*pi*(0:1:N)/N);

%     ne = ns + log(dn1/dn0)/log(1 + maxs);
% 
%     s = [];
%     
%     for i = 1:n
%     
%         s = [s; maxs*0.25*(1 + erf(6*(i-ns)/ws))*(1-erf(6*(i-ne)/we))];
%     
%     end
%     
%     f_ = zeros(size(s)); f_(1) = dn0;
%     
%     for k = 2:length(f_)
%     
%         f_(k) = f_(k-1)*(1 + s(k));
%         
%     end
%     
%     f = zeros(size(s)); f(1) = 0;
% 
%     for k = 2:length(f_)
% 
%         f(k) = f(k-1) + f_(k);
%         
%     end


end

