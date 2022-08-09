


def add_deq_args(parser):
    parser.add_argument('--wnorm', action='store_true', help="use weight normalization")
    parser.add_argument('--f_solver', default='anderson', type=str, choices=['anderson', 'broyden', 'naive_solver'],
                        help='forward solver to use (only anderson and broyden supported now)')
    parser.add_argument('--b_solver', default='broyden', type=str, choices=['anderson', 'broyden', 'naive_solver'],
                        help='backward solver to use')
    parser.add_argument('--f_thres', type=int, default=40, help='forward pass solver threshold')
    parser.add_argument('--b_thres', type=int, default=40, help='backward pass solver threshold')
    parser.add_argument('--f_eps', type=float, default=1e-3, help='forward pass solver stopping criterion')
    parser.add_argument('--b_eps', type=float, default=1e-3, help='backward pass solver stopping criterion')
    parser.add_argument('--f_stop_mode', type=str, default="abs", help="forward pass fixed-point convergence stop mode")
    parser.add_argument('--b_stop_mode', type=str, default="abs", help="backward pass fixed-point convergence stop mode")
    parser.add_argument('--eval_factor', type=float, default=1.5, help="factor to scale up the f_thres at test for better convergence.")
    parser.add_argument('--eval_f_thres', type=int, default=0, help="directly set the f_thres at test.")

    parser.add_argument('--indexing_core', action='store_true', help="use the indexing core implementation.")
    parser.add_argument('--ift', action='store_true', help="use implicit differentiation.")
    parser.add_argument('--safe_ift', action='store_true', help="use a safer function for IFT to avoid potential segment fault in older pytorch versions.")
    parser.add_argument('--n_losses', type=int, default=1, help="number of loss terms (uniform spaced, 1 + fixed point correction).")
    parser.add_argument('--indexing', type=int, nargs='+', default=[], help="indexing for fixed point correction.")
    parser.add_argument('--phantom_grad', type=int, nargs='+', default=[1], help="steps of Phantom Grad")
    parser.add_argument('--tau', type=float, default=1.0, help="damping factor for unrolled Phantom Grad")
    parser.add_argument('--sup_all', action='store_true', help="supervise all the trajectories by Phantom Grad.")

    parser.add_argument('--sradius_mode', action='store_true', help="monitor the spectral radius during validation")


