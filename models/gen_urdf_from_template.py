from jinja2 import Template

catch_ee_geom = "<visual> \
      <origin xyz=\"0 0 0\" rpy=\"0 0 0\" /> \
      <geometry> \
        <cylinder radius=\".04\" length=\".02\"/> \
      </geometry> \
      <material name=\"tip\"/>  \
    </visual> \
    <collision> \
      <origin xyz=\"0 0 0\" rpy=\"0 0 0\" /> \
      <geometry> \
        <cylinder radius=\".04\" length=\".02\"/> \
      </geometry> \
    </collision>"

# uam_0dof_com_middle
# arm_length = 0.15
# arm_length_geom = arm_length - 0.035
# arm_origin_geom = (arm_length_geom - 0.035) / 2
# com_from_vehicle = 0.04
# com_from_ee = (arm_length + 0.05) - com_from_vehicle

# uam_0dof_long_arm_com_middle
# arm_length = 0.35
# arm_length_geom = arm_length - 0.035
# arm_origin_geom = (arm_length_geom - 0.035) / 2
# com_from_vehicle = 0.04
# com_from_ee = (arm_length + 0.05) - com_from_vehicle

# uam_quadrotor
# arm_length = -0.05
# arm_length_geom = arm_length - 0.035
# arm_origin_geom = (arm_length_geom - 0.035) / 2
# com_from_vehicle = 0.0
# com_from_ee = (arm_length + 0.05) - com_from_vehicle
# add_catch_geom = False
# output_name = "uam_quadrotor.urdf"



arm_length = 0.15
arm_length_geom = arm_length - 0.035
arm_origin_geom = (arm_length_geom - 0.035) / 2
com_from_vehicle = 0.04
com_from_ee = (arm_length + 0.05) - com_from_vehicle
add_catch_geom = True
# output_name = "uam_0dof_com_middle_catching.urdf"
output_name = "uam_0dof_catching.urdf"



with open("uam_0dof_template.urdf", "r") as file:
    template = Template(file.read())

gripper_geom = catch_ee_geom if add_catch_geom else ""
params = {
    "arm_length": arm_length,
    "arm_length_geom": arm_length_geom,
    "arm_origin_geom": arm_origin_geom,
    "com_from_vehicle": com_from_vehicle,
    "com_from_ee": com_from_ee,
    "gripper_geom": gripper_geom
}

output = template.render(params)

with open(output_name, "w") as file:
    file.write(output)