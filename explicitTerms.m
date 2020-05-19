function [ rHat ] = explicitTerms( Lhat, Re, dt, Nhat, NhatOld, u, v )

    u = reshape(u, [], 1);
    v = reshape(v, [], 1);
    
    rHat.u = (speye(size(Lhat.u))/dt + 0.5*Lhat.u/Re)*u...
        - 1.5*Nhat.u + 0.5*NhatOld.u; 

    rHat.v = (speye(size(Lhat.v))/dt + 0.5*Lhat.v/Re)*v...
        - 1.5*Nhat.v + 0.5*NhatOld.v;
            
    rHat = [rHat.u; rHat.v];

end

