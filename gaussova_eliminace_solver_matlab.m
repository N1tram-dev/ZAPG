% GAUSSOVA ELIMINACE – MATLAB 
% Uložte jako gauss_solve_benchmark.m a spusťte (F5).
% Autor: Martin Lukáše
% Dost bylo překopírováno rovnou z pythonu.

clear; clc;

% --- TEST 3x3 (shodný se zdrojovým Python skriptem)
A = [  2,  1, -1;
       -3, -1,  2;
       -2,  1,  2 ];
b = [8; -11; -3];

[x, U, bb] = gauss_solve(A, b, true, true, false);
disp('Vrchní trojúhelníkový tvar U:');
disp(U);
disp('Změněná pravá strana b:');
disp(bb);
disp('Vektor x:');
disp(x);

% --- Benchmark & Plot (velikosti, počet opakování, pivotování)
[sizes, t_gauss, t_mldiv] = benchmark([10 100 300 500], 3, true);

figure;
plot(sizes, t_gauss, '-o', 'DisplayName', 'Gaussian elimination'); hold on;
plot(sizes, t_mldiv, '-o', 'DisplayName', 'A\\b');
xlabel('Maticová velikost n (n \times n)');
ylabel('průměrný čas řešení (s)');
title('Čas řešení vs. maticová velikost');
legend('Location','northwest');
grid on; box on;


%% ============ Lokální funkce ============
function [x, U, bb] = gauss_solve(A, b, pivot, return_intermediate, verbose)
% GAUSS_SOLVE   Řeší Ax=b pomocí Gaussovy eliminace se zpětnou substitucí.
% Nepoužívá vestavěné solvery (\, lu, rref, ...).
%
% Vstupy:
%   A  ... (n x n) čtvercová matice (double)
%   b  ... (n x 1) pravá strana
%   pivot (log.)  ... true = částečné pivotování (default = true)
%   return_intermediate (log.) ... pokud true, vrací i U a b (default = false)
%   verbose (log.) ... mezivýpisy po každém sloupci (default = false)
%
% Výstupy:
%   x  ... řešení (n x 1)
%   U  ... horní trojúhelníková matice po eliminaci
%   b ... pravá strana po eliminaci

    if nargin < 3 || isempty(pivot), pivot = true; end
    if nargin < 4 || isempty(return_intermediate), return_intermediate = false; end
    if nargin < 5 || isempty(verbose), verbose = false; end

    A = double(A);
    b = double(b(:));

    [n, m] = size(A);
    if n ~= m
        error('Matice musí být čtvercová.');
    end
    if numel(b) ~= n
        error('Rozměr b musí odpovídat počtu neznámých (n).');
    end

    U = A;
    bb = b; %bohužel program z nějakého důvodu nechtěl překousat jen b jak v pythonu.
    tol = 1e-12;

    % Dopředná eliminace
    for k = 1:(n-1)
        if pivot
            % index řádku s největší absolutní hodnotou v aktuálním sloupci
            [~, irel] = max(abs(U(k:n, k)));
            imax = k + irel - 1;
            if abs(U(imax, k)) < tol
                error('Matice je singulární nebo téměř singulární.');
            end
            if imax ~= k
                U([k, imax], :) = U([imax, k], :);
                bb([k, imax]) = bb([imax, k]);
            end
        else
            if abs(U(k, k)) < tol
                error('Pivot je nula; zapněte pivotování (pivot=true).');
            end
        end

        % Eliminace prvků pod pivotem
        for i = k+1:n
            m = U(i, k) / U(k, k);
            U(i, k:n) = U(i, k:n) - m * U(k, k:n);
            bb(i) = bb(i) - m * bb(k);
        end

        if verbose
            fprintf('Po eliminaci sloupce %d:\n', k);
            disp(U);
        end
    end

    % Zpětná substituce
    x = zeros(n, 1);
    for i = n:-1:1
        s = U(i, i+1:end) * x(i+1:end);
        x(i) = (bb(i) - s) / U(i, i);
    end

    if ~return_intermediate
        % Pokud U a b uživatel nechce, a přesto si je nevyžádal jako výstupy,
        % je to v MATLABu v pořádku – nebudou vráceny.
    end
end


function [A, b] = well_conditioned_random_system(n)
% WELL_CONDITIONED_RANDOM_SYSTEM   Vytvoří náhodnou matici.
% Inspirace z Python verze: A = randn + n*E pro diagonální dominanci.
    A = randn(n, n);
    A = A + n * eye(n);
    b = randn(n, 1);
end


function [sizes, t_gauss, t_mldiv] = benchmark(sizes, trials, pivot)
% BENCHMARK   Srovnání času gauss_solve vs. A\b pro různé velikosti matic.
%   sizes  ... vektor velikostí (např. [10 100 300 500])
%   trials ... kolikrát průměrovat pro každé n (např. 3)
%   pivot  ... logická hodnota, zda používat pivotování v gauss_solve
%
%   Výstupy: velikosti, průměrné časy pro gauss_solve a pro A\b

    if nargin < 3, pivot = true; end
    rng(42);  % reprodukovatelnost

    sizes = sizes(:)';
    t_gauss = zeros(size(sizes));
    t_mldiv = zeros(size(sizes));

    for idx = 1:numel(sizes)
        n = sizes(idx);
        tg = 0.0; tm = 0.0;

        for t = 1:trials
            [A, b] = well_conditioned_random_system(n);

            % Čas gauss_solve
            t0 = tic;
            gauss_solve(A, b, pivot, false, false);
            tg = tg + toc(t0);

            % Čas A\b
            t0 = tic;
            A\b; % výpočet bez ukládání
            tm = tm + toc(t0);
        end

        t_gauss(idx) = tg / trials;
        t_mldiv(idx) = tm / trials;
    end
end
