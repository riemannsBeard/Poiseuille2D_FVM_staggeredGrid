function [ Nhat, ua, va, ue, ve ] = convectionHat(grid, u, v, Nx, Ny, bc)

    u = reshape(u, Ny, Nx-1);
    v = reshape(v, Ny-1, Nx);   
    
    %% Condiciones de contorno (en los nodos de la malla externa)
    
    % u
    uN = grid.X*0;           
    uS = grid.X*0;               
    uW = bc.uW;         
    uE = u(:,end);
    
    % v
    vN = 0.5*(grid.X(2:end) + grid.X(1:end-1))*0;
    vS = 0.5*(grid.X(2:end) + grid.X(1:end-1))*0;
    vW = grid.Y'*0;
    vE = grid.Y'*0;

    %%
    
    Nhat.u = zeros(size(u));
    Nhat.v = zeros(size(v));
    
    % u
    u2 = u.^2;   
    u2c = zeros(Ny, Nx);
    u2c(:,1) = 0.5*(bc.uW.^2 + u2(:,1));
    u2c(:,2:end-1) = 0.5*(u2(:,2:end) + u2(:,1:end-1));
    u2c(:,end) = 0.5*(uE.^2 + u2(:,end));
    
    Nhat.u = diff(u2c')'./grid.dXp';

    % v
    v2 = v.^2;    
    v2c = zeros(Ny, Nx);
    v2c(end,:) = 0.5*(bc.vN.^2 + v2(end,:));
    v2c(2:end-1,:) = 0.5*(v2(2:end,:) + v2(1:end-1,:));
    v2c(1,:) = 0.5*(bc.vS.^2 + v2(1,:));
    
    Nhat.v = diff(v2c)./grid.dYp;
    
    % uv
    uv = 0.25*(u(2:end,:) + u(1:end-1,:)).*(v(:,2:end) + v(:,1:end-1));
    
    uvS = 0.5*bc.uS.*(bc.vS(2:end) + bc.vS(1:end-1));
    uvN = 0.5*bc.uN.*(bc.vN(2:end) + bc.vN(1:end-1));
    
    uvW = 0.5*bc.vW.*(bc.uW(2:end) + bc.uW(1:end-1));
    uvE = 0.5*bc.vE.*(uE(2:end) + uE(1:end-1));
    %uvE = 0.5*v(:,end).*(uE(2:end) + uE(1:end-1));
    
    Nhat.u = reshape(Nhat.u + diff([uvS; uv; uvN], 1, 1)./grid.dY, [], 1);
    Nhat.v = reshape(Nhat.v + diff([uvW uv uvE], 1, 2)./grid.dX', [], 1);

    %% Velocidades en la malla externa

    % Solucion en los vertices de las celdillas
    ue = [uW u uE];
    ue = [2*uS - ue(1,:); ue; 2*uN - ue(end,:)];
    
    ve = [vS; v; vN];
    ve = [2*vW - ve(:,1) ve 2*vE - ve(:,end)];
    %ve = [2*vW - ve(:,1) ve ve(:,end)];


    %% Solucion en los nodos correspondientes
    
    ua = 0.5*(ue(2:end,:) + ue(1:end-1,:));
    va = 0.5*(ve(:,2:end) + ve(:,1:end-1));
    
    
end

