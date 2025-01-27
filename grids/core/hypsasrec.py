from grids.core import sasrec

trial_params = dict(**sasrec.trial_params)

fixed_params = dict(
        **sasrec.fixed_params,
        c = 1,
        bias = True,
        geom = 'ball',
        model_save=False,
        pretrained=False,
        train_curv=True
)
