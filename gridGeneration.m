function [ grid, u, v, p ] = gridGeneration( Lx, Ly, Nx, Ny, alpha)

    grid.X = linspace(0, Lx, Nx+1);
    grid.Y = linspace(0, Ly, Ny+1);
        
    % grid.X = stretching(alpha.x, Nx);
    grid.Y = stretching(alpha.y, Ny);
 
    grid.dX = diff(grid.X)';
    grid.dY = diff(grid.Y)';

    grid.cellMin = min([grid.dX; grid.dY]);

    grid.Xp = 0.5*(grid.X(2:end) + grid.X(1:end-1));
    grid.Yp = 0.5*(grid.Y(2:end) + grid.Y(1:end-1));
    
    grid.dXp = diff(grid.Xp)';
    grid.dYp = diff(grid.Yp)';
    
    p = zeros(Ny, Nx);
    u = ones(Ny, Nx-1);
    v = zeros(Ny-1, Nx);

    grid.Xu = grid.X(2:end-1);
    grid.Yu = grid.Yp;

    grid.Xv = grid.Xp;
    grid.Yv = grid.Y(2:end-1);

    [grid.x, grid.y] = meshgrid(grid.X, grid.Y);
    [grid.xp, grid.yp] = meshgrid(grid.Xp, grid.Yp);
    [grid.xu, grid.yu] = meshgrid(grid.Xu, grid.Yu);
    [grid.xv, grid.yv] = meshgrid(grid.Xv, grid.Yv);
    
    grid.dx = diff(grid.x')';
    grid.dy = diff(grid.y);

    grid.dxp = diff(grid.xp')';
    grid.dyp = diff(grid.yp);
    

end

