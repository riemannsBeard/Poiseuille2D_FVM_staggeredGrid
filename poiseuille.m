clc
clear all
close all

%% Datos

Re = 250;
Nx = 512; % Celdillas en x
Ny = 64; % Celdillas en y
Lx = 20;
Ly = 1;
tf = 20;
tSampling = 0.5;
CFL = 0.1;

%% Staggered Grid Generation

alpha.x = 0.;
alpha.y = 0.125;

[grid, u, v, p] = gridGeneration(Lx, Ly, Nx, Ny, alpha);
dt = CFL*min(grid.cellMin^2*Re, grid.cellMin);

itSampling = floor(tSampling/dt);

%% Boundary conditions

bc.uS = zeros(1,Nx-1);
bc.uN = zeros(1,Nx-1);
bc.uE = zeros(Ny,1);
bc.uW = ones(Ny,1);

bc.vS = zeros(1,Nx);
bc.vN = zeros(1,Nx);
bc.vE = zeros(Ny-1,1);
bc.vW = zeros(Ny-1,1);


%% Grid detail
lx = 2;
l = grid.x < lx;
fig = figure;
plot(grid.x(l), grid.y(l), 'k.'), pbaspect([lx Ly 1])
set(gca, 'TickLabelInterpreter','latex', 'fontsize', 12)
xlabel('$l_x$', 'interpreter', 'latex', 'fontsize', 16)
ylabel('$L_y$', 'interpreter', 'latex', 'fontsize', 16)
printFigure, print(fig, 'pipeFlow_gridDetail', '-dpdf', '-r0');


%% Operators

[D, G, R, M, bc2_] = DGRM(grid, Nx, Ny);

Lhat = laplacianHat(Nx, Ny, grid);
L.L = M.hat*Lhat.L/R.R;

M.M = M.hat/R.R;
M.inv = inv(M.M);

Ahat = sparse(speye(size(Lhat.L))/dt - 0.5*Lhat.L/Re);
A = M.hat*Ahat/R.R;
dA = decomposition(A);

BN = dt*speye(size(M.M))/M.M + (0.5/Re)*dt*dt*(M.inv*L.L)*M.inv +...
    ((0.5/Re)^2)*(dt^3)*((M.inv*L.L)^2)*M.inv;

LHS = sparse(G.G'*BN*G.G);
dLHS = decomposition(LHS);

%% Simulation

uOld = u;
vOld = v;

t = 0;
k = 0;

% Preallocating of residual vectors
epsU = zeros(floor(tf/dt),1);
epsV = epsU;

tic
while t<= tf
    
    u = reshape(u, [], 1);
    v = reshape(v, [], 1);
        
    % Advective terms
    [NhatOld, ~, ~] = convectionHat(grid, uOld, vOld, Nx, Ny, bc);
    [Nhat, ua, va] = convectionHat(grid, u, v, Nx, Ny, bc);
    
    rnHat = explicitTerms(Lhat, Re, dt, Nhat, NhatOld, u, v);  
    rn = M.hat*rnHat;
        
    %% 1. Solve for intermediate velocity
       
    % BC's due to Laplacian
    bc1hat.u = Lhat.ux0*bc.uW + Lhat.uy1*bc.uN' + ...
        Lhat.uy0*bc.uS';
    bc1hat.v = Lhat.vx0*bc.vW + Lhat.vx1*bc.vE + Lhat.vy1*bc.vN' + ...
        Lhat.vy0*bc.vS';

    bc1 = M.hat*[bc1hat.u; bc1hat.v]/Re;
    
    % Flux calculation    
    q = dA\(rn + bc1);
    qu = q(1:Ny*(Nx-1));
    qv = q(Ny*(Nx-1)+1:end);

  
    %% 2. Solve the Poisson Equation
    
    % BC's due to Divergence
    bc2 = D.uW*(bc.uW.*grid.dY) + ...
        D.vS*(bc.vS'.*grid.dX) + D.vN*(bc.vN'.*grid.dX);

    RHS = G.G'*q + bc2;
    
    phi = dLHS\RHS;
    
    %% 3. Projection step
    
    q = q - BN*G.G*phi;
       
    vel = R.R\q;

    t = t + dt;
    k = k + 1;

    % Residuals
    epsU(k) = max(abs(u - vel(1:Ny*(Nx-1))));
    epsV(k) = max(abs(v - vel(Ny*(Nx-1)+1:end)));

    % Separation of velocity components
    u = vel(1:Ny*(Nx-1));    
    v = vel(Ny*(Nx-1)+1:end);
    phi = reshape(phi, Ny, Nx);    

    % Information
    fprintf(['t = ' num2str(t) '. Elapsed time: ' num2str(toc) 's \n']);
    fprintf(['Residuals u = ' num2str(epsU(k)) '.\n'...
        'Residuals v = ' num2str(epsV(k)) '\n \n']);    
    

    % On-the-fly plots (comment for speeding up the code)
    if (mod(k, itSampling) == 0)
        figure(1),
        
        ax = subplot(311);
        pcolor(grid.x, grid.y, ua), shading interp
        cmap = jet;
        colorbar, colormap(ax, cmap)
        set(gca, 'TickLabelInterpreter','latex', 'fontsize', 12)
        title('$u$', 'interpreter', 'latex', 'fontsize', 16)
        pbaspect([Lx Ly 1])
        
        ax = subplot(312);
        pcolor(grid.x, grid.y, va), shading interp
        cmap = jet;
        colorbar, colormap(ax, cmap)
        set(gca, 'TickLabelInterpreter','latex', 'fontsize', 12)
        title('$v$', 'interpreter', 'latex', 'fontsize', 16)
        pbaspect([Lx Ly 1])
        
        ax = subplot(313);
        contourf(grid.xp, grid.yp, phi), %shading interp,
        colorbar
%         cmap = jet;
%         colorbar, colormap(ax, cmap)
        set(gca, 'TickLabelInterpreter','latex', 'fontsize', 12)
        title('$\phi$', 'interpreter', 'latex', 'fontsize', 16)
        pbaspect([Lx Ly 1])
        
        drawnow
        
        figure(2),
        loglog(1:k, epsU(1:k), 1:k, epsV(1:k))
        set(gca, 'TickLabelInterpreter','latex', 'fontsize', 12)
        h = legend('$u$', '$v$');
        set(h, 'interpreter', 'latex', 'fontsize', 16)
        xlabel('$N$', 'interpreter', 'latex', 'fontsize', 16)
        ylabel('$\xi$', 'interpreter', 'latex', 'fontsize', 16)
        title('Residuals')
        drawnow
        
    end
    
end
toc

%% Plots

% Contours
fig = figure(1);

ax = subplot(311);
pcolor(grid.x, grid.y, ua), shading interp
cmap = jet;
colorbar, colormap(ax, cmap)
set(gca, 'TickLabelInterpreter','latex', 'fontsize', 12)
title('$u$', 'interpreter', 'latex', 'fontsize', 16)
pbaspect([Lx Ly 1])

ax = subplot(312);
pcolor(grid.x, grid.y, va), shading interp
cmap = jet;
colorbar, colormap(ax, cmap)
set(gca, 'TickLabelInterpreter','latex', 'fontsize', 12)
title('$v$', 'interpreter', 'latex', 'fontsize', 16)
pbaspect([Lx Ly 1])

ax = subplot(313);
contourf(grid.xp, grid.yp, phi), %shading interp,
colorbar
%         cmap = jet;
%         colorbar, colormap(ax, cmap)
set(gca, 'TickLabelInterpreter','latex', 'fontsize', 12)
title('$\phi$', 'interpreter', 'latex', 'fontsize', 16)
pbaspect([Lx Ly 1])

saveas(fig, 'pipeFlow', 'jpeg');

% Residuals
fig = figure(5);
loglog(1:k, epsU(1:k), 1:k, epsV(1:k))
set(gca, 'TickLabelInterpreter','latex', 'fontsize', 12)
h = legend('$u$', '$v$');
set(h, 'interpreter', 'latex', 'fontsize', 16)
xlabel('$N$', 'interpreter', 'latex', 'fontsize', 16)
ylabel('$\xi$', 'interpreter', 'latex', 'fontsize', 16)
title('Residuals')
printFigure, print(fig, 'pipeFlow_res', '-dpdf', '-r0');


%% Validation

% Numerical solution
x = Lx-1; % Sample point
yq = linspace(0, Ly, 256);
xq = yq*0 + x;
uNum = interp2(grid.x, grid.y, ua, xq, yq);

% Theoretical solution
Q = sum(bc.uW.*grid.dY);
uTeo = 6*Q*yq.*(Ly-yq)/Ly^3;

% Comparison
fig = figure(4);
plot(uNum, yq), hold on,
set(gca, 'TickLabelInterpreter','latex', 'fontsize', 12)
xlabel('$y$', 'interpreter', 'latex', 'fontsize', 16)
ylabel('$u$', 'interpreter', 'latex', 'fontsize', 16)
plot(uTeo, yq, '--')
h = legend('Num', 'Teo');
set(h, 'interpreter', 'latex');
hold off
printFigure, print(fig, 'pipeFlow_uProfile', '-dpdf', '-r0');

