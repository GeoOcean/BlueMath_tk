import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

def SS_Distributions(xds_ss,xds_kma):
    
    clusters=np.where(np.unique(xds_kma.bmus)>=0)[0]
    n_clusters=len(clusters)
    n_rows=int(np.sqrt(n_clusters+1))
    n_cols=n_rows
    
    fig = plt.figure(figsize=[20,12])
    gs = gridspec.GridSpec(n_rows, n_cols, wspace=0.0, hspace=0.0)
    grid_row = 0
    grid_col = 0
    for ic in clusters:
        # data mean
        pos_cluster = np.where(xds_kma.bmus==ic)[0][:]
        ax = plt.subplot(gs[grid_row, grid_col])
        plt.hist(xds_ss.ss[pos_cluster],range=[-0.30, 0.30],bins=40,color='indigo',histtype='stepfilled',density=True, alpha=0.5)
        ax.set_xticks([])
        ax.set_yticks([])

        if grid_row == n_rows-1:
            ax.set_yticks([])
            ax.set_xticks(np.arange(-0.30,0.31, step=0.1))
            ax.set_xlabel('SS',fontsize=14)
        else:
            ax.set_xticks([])
            ax.set_yticks([]) 

        grid_row += 1
        if grid_row >= n_rows:
            grid_row = 0
            grid_col += 1
            
    fig = plt.figure(figsize=[20,12])
    gs = gridspec.GridSpec(n_rows, n_cols, wspace=0.0, hspace=0.0)
    grid_row = 0
    grid_col = 0
    for ic in clusters:
        # data mean
        pos_cluster = np.where(xds_kma.bmus==ic)[0][:]
        ax = plt.subplot(gs[grid_row, grid_col])
        plt.hist(xds_ss.Dwind[pos_cluster],range=[0,360],bins=50,color='darkorange',histtype='stepfilled', alpha=0.5)
        ax.set_xticks([])
        ax.set_yticks([])

        if grid_row == n_rows-1:
            ax.set_yticks([])
            ax.set_xticks(np.arange(0,361, step=90))
            ax.set_xlabel('WDir',fontsize=14)
        else:
            ax.set_xticks([])
            ax.set_yticks([])

        grid_row += 1
        if grid_row >= n_rows:
            grid_row = 0
            grid_col += 1
    
    fig = plt.figure(figsize=[20,12])
    gs = gridspec.GridSpec(n_rows, n_cols, wspace=0.0, hspace=0.0)
    grid_row = 0
    grid_col = 0
    for ic in clusters:
        # data mean
        pos_cluster = np.where(xds_kma.bmus==ic)[0][:]
        ax = plt.subplot(gs[grid_row, grid_col])
        plt.hist(xds_ss.wind[pos_cluster],range=[2,20],bins=50,color='peru',histtype='stepfilled', alpha=0.5)
        if grid_row == n_rows-1:
            ax.set_yticks([])
            ax.set_xticks(np.arange(5,21, step=5))
            ax.set_xlabel('WSpeed',fontsize=14)
        else:
            ax.set_xticks([])
            ax.set_yticks([])

        grid_row += 1
        if grid_row >= n_rows:
            grid_row = 0
            grid_col += 1