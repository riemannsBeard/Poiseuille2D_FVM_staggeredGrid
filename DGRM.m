function [ D, G, R, M, bc2] = DGRM( grid, Nx, Ny )
% Funcion que devuelve los operadores matriciales G, D, R y M

    % Vectores unitarios auxiliares
    ex = ones(Nx+2, 1);
    ey = ones(Ny+2, 1);

    %% OPERADOR GRADIENTE
    % Se aplica a la presion
        
    % Operador derivada para una fila y una columna
    g.x = spdiags([-ex, ex], [0, 1], Nx-1, Nx);
    g.y = spdiags([-ey, ey], [0, 1], Ny-1, Ny);
           
    %% OPERADOR DIVERGENCIA
    % Se aplica a las velocidades
    d.x = -g.x';
    d.y = -g.y';
    
    % Account for Neumann BC
    d.x(end,end) = 0;

    D.x = kron(d.x, speye(Ny));
    D.y = kron(speye(Nx), d.y);

    D.D = [D.x D.y];
    
    G.G = -D.D';
    
    bcW = zeros(Nx,1); bcW(1) = 1;
    bcE = zeros(Nx,1); bcE(end) = -1;
    bcN = zeros(Ny,1); bcN(1) = 1;
    bcS = zeros(Ny,1); bcS(end) = -1;
    
    bc2.uW = kron(bcW, grid.dY);
    bc2.uE = kron(grid.dY, bcE);
    bc2.vN = kron(bcN, grid.dX);
    bc2.vS = kron(bcS, grid.dY);
    
    % Condiciones de contorno

    d.uW = spdiags(ex, 0, Nx, 1);
    D.uW = kron(d.uW, speye(Ny));
    
    d.uE = spdiags(-ex, -Nx+1, Nx, 1);
    D.uE = kron(d.uE, speye(Ny));
    
    d.vN = spdiags(ey, 0, Ny, 1);
    D.vN = kron(speye(Nx), d.vN);
    
    d.vS = spdiags(ey, -Ny+1, Ny, 1);
    D.vS = kron(speye(Nx), d.vS);
        
    %% MATRIZ DE FLUJO
    
    dyj = spdiags(grid.dY, 0, Ny, Ny);
    dxi = spdiags(grid.dX, 0, Nx, Nx);
    
    R.u = kron(speye(Nx-1), dyj);
    R.v = kron(dxi, speye(Ny-1));

    R.R = blkdiag(R.u, R.v);
           
    %% MATRIZ DE MASA
    
    ix = [0.75; ones(Nx-2,1); 1];
    iy = [0.75; ones(Ny-2,1); 0.75];

    Ix = spdiags(ix, 0, Nx, Nx);
    Iy = spdiags(iy, 0, Ny, Ny);
    
    Dxp = spdiags(grid.dXp, 0, Nx-1, Nx-1);
    Dyp = spdiags(grid.dYp, 0, Ny-1, Ny-1);

    % Mhat
    Mhat.u = kron(Dxp, Iy);
    Mhat.v = kron(Ix, Dyp);

    M.hat = blkdiag(Mhat.u, Mhat.v);
    
    % M
    M.M = R.R\M.hat;
    
end

