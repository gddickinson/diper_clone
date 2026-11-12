"""Web interface for DiPer (python run_web.py -> http://127.0.0.1:8156).

Contents:
    app.py       FastAPI app: front end in static/, JSON API under /api, STATIC_ENDPOINTS for the static site
    analysis.py  bridge from uploads to the repo's analysis functions (load_data, split_trajectories, Speed, Dir_Ratio,
                 MSD, Autocorrel, Autocorrel_3D, Vel_Cor, Plot_At_Origin, Sparse_Data)
    static/      plain ES-module front end; diper_engine.js is the JS port used on the GitHub Pages site
"""
