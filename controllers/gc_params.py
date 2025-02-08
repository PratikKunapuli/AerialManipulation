"""
Dictionary of parameters for the GC controller. 
Keys are the task, and values are dictionary of parameters for the GC controller.
"""
gc_params_dict = {
    "Isaac-Crazyflie-Hover-v0" : {
        "log_dir": "./crazyflie_baseline_gc/",
        "controller_params": {
            "kp_pos_gain_xy": 6.5,
            "kp_pos_gain_z": 15.0,
            "kd_pos_gain_xy": 4.0,
            "kd_pos_gain_z": 9.0,
            "kp_att_gain_xy": 544.0,
            "kp_att_gain_z": 544.0,
            "kd_att_gain_xy": 46.64,
            "kd_att_gain_z": 46.64,
        },
    },
}