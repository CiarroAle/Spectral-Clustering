
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%% PRINCIPAL CODE WITHOUT OPTIONAL POINTS %%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

 %% Adjacency matrix construction
function W_knn = knn_sim_graph(Sim, k)
    % INPUT:
    % Sim: symmetric similarity matrix
    % k: number of nearest neighbors
    % OUTPUT:
    % W_knn: adjacency matrix
    % initialization of n and W_knn
    n = size(Sim, 1);
    W_knn = zeros(n,n);

    % construction of W_knn
    for i = 1:n
        [~, indices] = sort(Sim(i,:), 'descend');
        neighbors = indices(2:k+1);
        W_knn(i, neighbors) = Sim(i, neighbors);
    end

    % W_knn must be symmetric
    W_knn = max(W_knn, W_knn');
end

%% The code starts
% Files to analyze
files = {'Circle.mat', 'Spiral.mat'};

% Parameters
k_values = [10,20,40];
sigma = 1;

% iteration on k
for f = 1:length(files)
    % file selection
    file = files{f};
    
    disp(['Processing file: ',file])
    
    data = load(file);
    X = data.X(:,1:2);

    % Similarity matrix construction
    Sim = exp((-squareform(pdist(X).^2))/(2*sigma));
    
    % iteration on k
    for k=k_values

        %% Exercise 1: Adjacency matrix

        W_knn = knn_sim_graph(Sim, k);
        
        %% Exercise 2: Laplacian matrix

        % Degree matrix construction
        D = sparse(diag(sum(W_knn, 2)));

        % Store W_knn in sparse format
        W_knn = sparse(W_knn);
        
        % Laplacian matrix construction
        L = sparse(D - W_knn);
        
        %% Exercise 3: Connected components with laplacian
         
        m = 10;  
        tolerance = 1e-3;
        smallest_eigenvalues = eigs(L,k,'smallestabs');
        num_components_L = sum(abs(smallest_eigenvalues) < tolerance);
        disp(['With the Laplacian matrix, we found ', num2str(num_components_L), ' connected components for k = ', num2str(k)] )
        
        %% Exercise 4 - 5: smallest eigenvalues and corresponding eigenvectors

        [V, A] = eigs(L, 3, 'smallestabs');
        Lambda = diag(A);
        disp(Lambda);
        
        %% Exercise 6: Spectral clustering
        
        M = 3;
        [labels, centroids] = kmeans(V, M, 'MaxIter', 1000);

        %% Exercise 7
        num_clusters = M; % to visualize better

        % Assign points according to the labels
        clusters = cell(num_clusters, 1); % creates a cell for each cluster
        for i = 1:num_clusters
            clusters{i} = X(labels == i, :);    % points of X for the i-cluster
        end

        %% Exercise 8
        figure;
        hold on;
        colors = lines(M); 
        
        % for each cluster, show the points
        for i = 1:num_clusters
            scatter(clusters{i}(:, 1), clusters{i}(:, 2), 50, 'MarkerEdgeColor', colors(i, :), 'DisplayName', ['Cluster ' num2str(i)]);
        end
        
        title('Spectral clustering for k = ', num2str(k));
        xlabel('X1');
        ylabel('X2');
        legend('show');
        hold off;

    end

    %% Exercise 9.1: Clustering with DBSCAN (parameters A)
    
    % Parameters A
    epsilon = 3;
    min_pts = 5;

    % DBSCAN application
    labels_DBSCAN = dbscan(X, epsilon, min_pts);

    % Number of cluster identified
    num_clusters_DBSCAN = max(labels_DBSCAN);
    disp(['DBSCAN found ', num2str(num_clusters_DBSCAN), 'clusters']);

    % Assign points according to the labels
    clusters_DBSCAN = cell(num_clusters_DBSCAN, 1);
    for i = 1:num_clusters_DBSCAN
        clusters_DBSCAN{i} = X(labels_DBSCAN == i, :);
    end

    % Clusters visualization
    figure;
    hold on;
    colors_2 = lines(num_clusters_DBSCAN);

    for i = 1:num_clusters_DBSCAN
        scatter(clusters_DBSCAN{i}(:, 1), clusters_DBSCAN{i}(:, 2), 50, 'MarkerEdgeColor', colors_2(i, :), ...
            'DisplayName', ['Cluster ', num2str(i)]);
    end

    noise_points = X(labels_DBSCAN == -1, :);
    if ~isempty(noise_points)
        scatter(noise_points(:,1), noise_points(:, 2), 50, 'k', 'filled', 'DisplayName', 'Noise');
    end

    title('DBSCAN Clustering with Parameters A');
    xlabel('X1');
    ylabel('X2');
    legend('show');
    hold off;

    %% Exercise 9.2 (Clustering with k-means) --> it will not be replicate in the other codes because the results will be the same (clearly).

    [idx, C] = kmeans(X, num_clusters, 'Replicates', 10);
    
    colors = lines(k); 
    
    % Clusters visualization
    figure;
    gscatter(X(:,1), X(:,2), idx, colors, 'o', 8);
    hold on;

    plot(C(:,1), C(:,2), 'kx', 'MarkerSize', 15, 'LineWidth', 3); 

    title(['Clustering con k-means, Parametri A, e k: ', num2str(k)]);
    xlabel('X1');
    ylabel('X2');

    legend_labels = arrayfun(@(i) ['Cluster ' num2str(i)], 1:k, 'UniformOutput', false);
    legend([legend_labels, {'Centroids'}], 'Location', 'best');
    
    grid on;
    hold off;

end

clear;



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%% CODE WITH OPTIONAL 2 AND NO INVERSE POWER METHOD %%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%% The code starts
% Files to analyze
files = { 'Circle.mat', 'Spiral.mat'};

% Parameters
k_values = [10,20,40];
sigma = 1;

for f = 1:length(files)
    % file selection
    file = files{f};
    
    disp(['Processing file: ',file])
    
    data = load(file);
    X = data.X(:,1:2);
    
    % Similarity matrix construction
    Sim = exp((-squareform(pdist(X).^2))/(2*sigma));

    % iteration on k 
    for k=k_values

        %% Exercise 1
        % knn_sim_graph application
        W_knn = knn_sim_graph(Sim, k);
        
        %% Exercise 2: Laplacian matrix

        % Degree matrix construction
        D = sparse(diag(sum(W_knn, 2)));

        % Store W_knn in sparse format
        W_knn = sparse(W_knn);
        
        % Laplacian matrix construction
        L = sparse(D - W_knn);

        %% Optional 2: L normalized symmetric

        L_sym = eye(length(L(1,:))) - (D^(-1/2)) * W_knn * (D^(-1/2));
        
        %% Exercise 3: Connected components with laplacian
        
        m = 10; 
        tolerance = 1e-3;
        smallest_eigenvalues = eigs(L_sym,k,'smallestabs');
        num_components_L = sum(abs(smallest_eigenvalues) < tolerance);
        disp(['With the Laplacian matrix, we found ', num2str(num_components_L), ' connected components for k = ', num2str(k)] )

        %% Exercises 4 - 5: smallest eigenvalues and corresponding eigenvectors

        [V, D] = eigs(L, 3, 'smallestabs');
        Lambda = diag(D);
        disp(Lambda);

        %% Exercise 6
        % k-means application
        M = 3;
        [labels, centroids] = kmeans(V, M, 'MaxIter', 10000);

        %% Exercise 7
        num_clusters = M; 

        % Assign points according to the labels
        clusters = cell(num_clusters, 1); % creates a cell for each cluster
        for i = 1:num_clusters
            clusters{i} = X(labels == i, :);    % points of X for the i-cluster
        end        

        %% Exercise 8
        figure;
        hold on;
        colors = lines(M); 
        
        % for each cluster, show the points
        for i = 1:num_clusters
            scatter(clusters{i}(:, 1), clusters{i}(:, 2), 50, 'MarkerEdgeColor', colors(i, :), 'DisplayName', ['Cluster ' num2str(i)]);
        end
        
        title('Spectral Clustering with k = ', num2str(k));
        xlabel('X1');
        ylabel('X2');
        legend('show');
        hold off;
    end

    %% Exercise 9.1: Cluster with DBSCAN, Parameters B
    
    % Parameters B
    epsilon = 1;
    min_pts = 5;

    % DBSCAN application
    labels_DBSCAN = dbscan(X, epsilon, min_pts);

    % Number of cluster identified
    num_clusters_DBSCAN = max(labels_DBSCAN);
    disp(['DBSCAN found ', num2str(num_clusters_DBSCAN), 'clusters for k = ', num2str(k)]);

    % Assign points according to the labels
    clusters_DBSCAN = cell(num_clusters_DBSCAN, 1);
    for i = 1:num_clusters_DBSCAN
        clusters_DBSCAN{i} = X(labels_DBSCAN == i, :);
    end

    % Clusters visualization
    figure;
    hold on;
    colors_2 = lines(num_clusters_DBSCAN);

    for i = 1:num_clusters_DBSCAN
        scatter(clusters_DBSCAN{i}(:, 1), clusters_DBSCAN{i}(:, 2), 50, 'MarkerEdgeColor', colors_2(i, :), ...
            'DisplayName', ['Cluster ', num2str(i)]);
    end

    noise_points = X(labels_DBSCAN == -1, :);
    if ~isempty(noise_points)
        scatter(noise_points(:,1), noise_points(:, 2), 50, 'k', 'filled', 'DisplayName', 'Noise');
    end

    title(['DBSCAN Clustering for k = ', num2str(k)]);
    xlabel('X1');
    ylabel('X2');
    legend('show');
    hold off;
end

clear;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%% INVERSE POWER METHOD OPTIONAL 3 %%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

warning('off', 'all');


%% Inverse power method
function [Lambda, U] = inverse_power_method(A, M, tol, max_iter)
    % INPUT:
    % A: initial matrix;
    % M: number of smallest eigenvalues to find with the method;
    % tol: the convergence tolerance;
    % max_iter: maximum number of iterations.
    % OUTPUT:
    % Lambda: a vector of the smallest M eigenvalues of A, sorted in
    % ascending order
    % U: matrix n x M, where each column is the eigenvector corresponding
    % to one of the smallest eigenvalues.

    % initialization
    n = size(A,1);
    Lambda = zeros(M,1);
    U = zeros(n, M);
    
    % iterative eigenvalue calculation
    for i = 1:M
        x = rand(n, 1); 
        x = x / norm(x); 
        lambda_old = 0;

        for iter=1:max_iter
            v = A \ x;                           % Solve Av=x efficiently
            lambda = x' * v;                     % Rayleigh quotient
            x = v / norm(v,2);                   % normalize v
            if abs(lambda - lambda_old) < tol    % check for the tolerance
                break
            end
            lambda_old = lambda;                 % update lambda_old
        end

        % store the results
        Lambda(i) = 1 / lambda;
        U(:,i) = x;


        % Deflation
        A = A \ eye(size(A)) - lambda * (x * x');
        A = A \ eye(size(A));
    end
            
end

%% The code starts
% Files to analyze
files = { 'Circle.mat', 'Spiral.mat'};

% Parameters
k_values = [10,20,40];
sigma = 1;

for f = 1:length(files)
    % file selection
    file = files{f};
    
    disp(['Processing file: ',file])
    
    data = load(file);
    X = data.X(:,1:2);
    
    % Similarity matrix construction
    Sim = exp((-squareform(pdist(X).^2))/(2*sigma));

    % iteration on k 
    for k=k_values
        %% Exercise 1: Adjacency matrix

        W_knn = knn_sim_graph(Sim, k);
        
        %% Exercise 2: Laplacian matrix

        % Degree matrix construction
        D = sparse(diag(sum(W_knn, 2)));

        % Store W_knn in sparse format
        W_knn = sparse(W_knn);
        
        % Laplacian matrix construction
        L = sparse(D - W_knn);
        
        %% Exercise 3: Connected components with laplacian

        m = 10; 
        tolerance = 1e-3;
        smallest_eigenvalues = eigs(L,k,'smallestabs');
        num_components_L = sum(abs(smallest_eigenvalues) < tolerance);
        disp(['With the Laplacian matrix, we found ', num2str(num_components_L), ' connected components for k = ', num2str(k)] )

        %% Exercises 4 (with optional 3) - 5: inverse power method

        M = 3;
        tol = 1e-6;
        max_iter = 1000;

        [Lambda, U] = inverse_power_method(L, M, tol, max_iter);
        disp(Lambda);
        
        
        %% Exercise 6: k-means application

        [labels, centroids] = kmeans(U, M, 'MaxIter', 10000);

        %% Exercise 7
        num_clusters = M; % to visualize better

        % Assign points according to the labels
        clusters = cell(num_clusters, 1); % creates a cell for each cluster
        for i = 1:num_clusters
            clusters{i} = X(labels == i, :);    % points of X for the i-cluster
        end

        %% Exercise 8
        figure;
        hold on;
        colors = lines(M); 
        
        % for each cluster, show the points
        for i = 1:num_clusters
            scatter(clusters{i}(:, 1), clusters{i}(:, 2), 50, 'MarkerEdgeColor', colors(i, :), 'DisplayName', ['Cluster ' num2str(i)]);
        end
        
        title('Cluster dei punti X con colori differenti con Spectral-clustering per k = ', num2str(k));
        xlabel('X1');
        ylabel('X2');
        legend('show');
        hold off;

    end

    %% Exercise 9.1: Cluster with DBSCAN, Parameters C
    
    % Parameters C
    epsilon = 1;
    min_pts = 10;

    % DBSCAN application
    labels_DBSCAN = dbscan(X, epsilon, min_pts);

    % Number of cluster identified
    num_clusters_DBSCAN = max(labels_DBSCAN);
    disp(['DBSCAN found ', num2str(num_clusters_DBSCAN), 'clusters ']);

    % Assign points according to the labels
    clusters_DBSCAN = cell(num_clusters_DBSCAN, 1);
    for i = 1:num_clusters_DBSCAN
        clusters_DBSCAN{i} = X(labels_DBSCAN == i, :);
    end

    % Clusters visualization
    figure;
    hold on;
    colors_2 = lines(num_clusters_DBSCAN);

    for i = 1:num_clusters_DBSCAN
        scatter(clusters_DBSCAN{i}(:, 1), clusters_DBSCAN{i}(:, 2), 50, 'MarkerEdgeColor', colors_2(i, :), ...
            'DisplayName', ['Cluster ', num2str(i)]);
    end

    noise_points = X(labels_DBSCAN == -1, :);
    if ~isempty(noise_points)
        scatter(noise_points(:,1), noise_points(:, 2), 50, 'k', 'filled', 'DisplayName', 'Noise');
    end

    title('DBSCAN Clustering with Parameters C');
    xlabel('X1');
    ylabel('X2');
    legend('show');
    hold off;

end

clear;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%% OPTIONAL 3 WITH L NORMALIZED %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%% The code starts
% Files to analyze
files = { 'Circle.mat', 'Spiral.mat'};

% Parameters
k_values = [10,20,40];
sigma = 1;

for f = 1:length(files)
    % file selection
    file = files{f};
    
    disp(['Processing file: ',file])
    
    data = load(file);
    X = data.X(:,1:2);

    % Similarity matrix construction
    Sim = exp((-squareform(pdist(X).^2))/(2*sigma));

    % iteration on k 
    for k=k_values
        %% Exercise 1: Adjacency matrix

        W_knn = knn_sim_graph(Sim, k);
        
        %% Exercise 2: Laplacian matrix

        % Degree matrix construction
        D = sparse(diag(sum(W_knn, 2)));

        % Store W_knn in sparse format
        W_knn = sparse(W_knn);
        
        % Laplacian matrix construction
        L = sparse(D - W_knn);

        %% Optional 2: L normalized symmetric

        L_sym = eye(length(L(1,:))) - (D^(-1/2)) * W_knn * (D^(-1/2));
        
        %% Exercise 3: Connected components with laplacian
        
        m = 10; 
        tolerance = 1e-3;
        smallest_eigenvalues = eigs(L_sym,k,'smallestabs');
        num_components_L = sum(abs(smallest_eigenvalues) < tolerance);
        disp(['With the Laplacian matrix, we found ', num2str(num_components_L), ' connected components for k = ', num2str(k)] )

        %% Exercises 4 (with optional 3) - 5: inverse power method

        M = 3;
        tol = 1e-6;
        max_iter = 1000;

        [Lambda, U] = inverse_power_method(L_sym, M, tol, max_iter);
        disp(Lambda);
        
        %% Exercise 6: k-means application

        [labels, centroids] = kmeans(U, M, 'MaxIter', 10000);

        %% Exercise 7
        num_clusters = M; % to visualize better

        % Assign points according to the labels
        clusters = cell(num_clusters, 1); % creates a cell for each cluster
        for i = 1:num_clusters
            clusters{i} = X(labels == i, :);    % points of X for the i-cluster
        end

        %% Exercise 8
        figure;
        hold on;
        colors = lines(M); 
        
        % for each cluster, show the points
        for i = 1:num_clusters
            scatter(clusters{i}(:, 1), clusters{i}(:, 2), 50, 'MarkerEdgeColor', colors(i, :), 'DisplayName', ['Cluster ' num2str(i)]);
        end
        
        title('Cluster dei punti X con colori differenti con Spectral-clustering per k = ', num2str(k));
        xlabel('X1');
        ylabel('X2');
        legend('show');
        hold off;

    end

    %% Exercise 9.1: Cluster with DBSCAN, Parameters D
    
    % Parameters D
    epsilon = 0.5;
    min_pts = 5;

    % DBSCAN application
    labels_DBSCAN = dbscan(X, epsilon, min_pts);

    % Number of cluster identified
    num_clusters_DBSCAN = max(labels_DBSCAN);
    disp(['DBSCAN found ', num2str(num_clusters_DBSCAN), ' clusters ']);

    % Assign points according to the labels
    clusters_DBSCAN = cell(num_clusters_DBSCAN, 1);
    for i = 1:num_clusters_DBSCAN
        clusters_DBSCAN{i} = X(labels_DBSCAN == i, :);
    end

    % Clusters visualization
    figure;
    hold on;
    colors_2 = lines(num_clusters_DBSCAN);

    for i = 1:num_clusters_DBSCAN
        scatter(clusters_DBSCAN{i}(:, 1), clusters_DBSCAN{i}(:, 2), 50, 'MarkerEdgeColor', colors_2(i, :), ...
            'DisplayName', ['Cluster ', num2str(i)]);
    end

    noise_points = X(labels_DBSCAN == -1, :);
    if ~isempty(noise_points)
        scatter(noise_points(:,1), noise_points(:, 2), 50, 'k', 'filled', 'DisplayName', 'Noise');
    end

    title('DBSCAN Clustering for Parameters D');
    xlabel('X1');
    ylabel('X2');
    legend('show');
    hold off;

end

clear;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%% 3D DATASET AND OPTIONAL 3 %%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%% The code starts

data = readtable('clustering_data.csv');
x = data.x;
y = data.y;
z = data.z;

% Points visualization
scatter3(x, y, z, 'filled');
xlabel('x');
ylabel('y'); 
zlabel('z'); 
grid on; 
title('Points visualization');

% Parameters
k_values = [10,20,40];
sigma = 1;

disp('Processing file "clustering_data"');

X = [x, y, z];


% Similarity matrix construction
Sim = exp((-squareform(pdist(X).^2))/(2*sigma));

for k=k_values
    %% Adjacency matrix
    W_knn = knn_sim_graph(Sim, k);

    D = sparse(diag(sum(W_knn, 2)));

    W_knn = sparse(W_knn);

    L = sparse(D - W_knn);

    %% Inverse power method
    M = 6;
    tol = 1e-6;
    max_iter = 1000;
    
    [Lambda, U] = inverse_power_method(L, M, tol, max_iter);

    %% Spectral clustering

    [labels, centroids] = kmeans(U, M, 'MaxIter', 1000);

    % Assign points according to the labels
    clusters = cell(M, 1); % creates a cell for each cluster
    for i = 1:M
        clusters{i} = X(labels == i, :);    % points of X for the i-cluster
    end

    % Clusters visualization
    figure;
    hold on;
    colors = lines(M);
    
    for i = 1:M
        scatter3(clusters{i}(:,1), clusters{i}(:,2), clusters{i}(:,3), 36, colors(i,:), 'o', 'LineWidth', 0.5);
    end
    
    xlabel('X');
    ylabel('Y');
    zlabel('Z');
    title(['Spectral clustering with k: ', num2str(k)]);
    
    hold off;

    
end

%% Exercise 9.2 (Clustering with k-means) 
num_clusters = 6;

[idx, C] = kmeans(X, num_clusters, 'Replicates', 10);

% Visualize the results
figure;
hold on
colors = lines(num_clusters); 
for i = 1:num_clusters
    cluster_points = X(idx == i, :); 
    scatter3(cluster_points(:, 1), cluster_points(:, 2), cluster_points(:, 3), 50, colors(i, :), 'o', 'DisplayName', ['Cluster ', num2str(i)]);
             
end
scatter3(C(:, 1), C(:, 2), C(:, 3), 200, 'k', 'x', 'LineWidth', 2, 'DisplayName', 'Centroids');
title(['Clustering with k-means, Number of Clusters: ', num2str(num_clusters)]);
xlabel('X1');
ylabel('X2');
zlabel('X3');
legend('show');
grid on;
hold off;



%% DBSCAN 
% Parameters
epsilon = 5;
min_pts = 10;

% DBSCAN application
labels_DBSCAN = dbscan(X, epsilon, min_pts);

% Number of cluster identified
num_clusters_DBSCAN = max(labels_DBSCAN);
disp(['DBSCAN found ', num2str(num_clusters_DBSCAN), 'for 3D dataset' ]);

% Assign points according to the labels
clusters_DBSCAN = cell(num_clusters_DBSCAN, 1);
for i = 1:num_clusters_DBSCAN
    clusters_DBSCAN{i} = X(labels_DBSCAN == i, :);
end

% Clusters visualization
figure;
hold on;
colors_2 = lines(num_clusters_DBSCAN);
for i = 1:num_clusters_DBSCAN
    scatter3(clusters_DBSCAN{i}(:, 1), clusters_DBSCAN{i}(:, 2), clusters_DBSCAN{i}(:,3), 50, 'MarkerEdgeColor', colors_2(i, :), ...
        'DisplayName', ['Cluster ', num2str(i)]);
end
noise_points = X(labels_DBSCAN == -1, :);
if ~isempty(noise_points)
    scatter3(noise_points(:,1), noise_points(:, 2), noise_points(:,3), 50, 'k', 'filled', 'DisplayName', 'Noise');
end
title('DBSCAN Clustering for 3D dataset ');
xlabel('X1');
ylabel('X2');
zlabel('X3')
legend('show');
hold off;


