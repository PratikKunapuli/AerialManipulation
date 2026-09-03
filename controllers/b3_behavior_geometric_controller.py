import math

import torch


class B3BehaviorGeometricController:
    """
    Behavior-faithful B3 Mellinger / SE(3) controller.

    Outer loop:
        Replicates the Crazyflie Mellinger position controller in SI units
        using the real B3 gains.

    Inner loop:
        Clean continuous-time SO(3) attitude controller in SI units.

    Plant interface:
        Direct body-frame collective thrust + body torque.

    This intentionally does NOT model:
        - estimator
        - BMI055
        - radio delay
        - PWM
        - power distribution
        - motor lag
        - firmware rate-derivative discretization

    The goal is faithful closed-loop geometric-controller behavior,
    not a digital twin.
    """

    def __init__(
        self,
        device,
        dtype=torch.float32,
        outer_hz=100.0,
        inner_hz=500.0,
    ):
        self.device = torch.device(device)
        self.dtype = dtype

        self.outer_hz = float(outer_hz)
        self.inner_hz = float(inner_hz)

        self.outer_div = int(
            round(self.inner_hz / self.outer_hz)
        )

        if self.outer_div < 1:
            raise ValueError("Invalid outer/inner controller rates")

        # ============================================================
        # REAL B3 PARAMETERS
        # ============================================================

        self.mass = torch.tensor(
            0.046,
            device=self.device,
            dtype=self.dtype,
        )

        self.J = torch.tensor(
            [
                2.4255e-05,
                1.8650e-05,
                3.9300e-05,
            ],
            device=self.device,
            dtype=self.dtype,
        )

        self.gravity = torch.tensor(
            [0.0, 0.0, -9.81],
            device=self.device,
            dtype=self.dtype,
        )

        # Real B3 Mellinger translational gains.
        self.kp = torch.tensor(
            [0.4, 0.4, 1.25],
            device=self.device,
            dtype=self.dtype,
        )

        self.kd = torch.tensor(
            [0.2, 0.2, 0.4],
            device=self.device,
            dtype=self.dtype,
        )

        self.ki = torch.tensor(
            [0.05, 0.05, 0.05],
            device=self.device,
            dtype=self.dtype,
        )

        self.int_pos_limit = torch.tensor(
            [2.0, 2.0, 0.4],
            device=self.device,
            dtype=self.dtype,
        )

        # ============================================================
        # BEHAVIORAL ENVELOPE
        #
        # Real B3 repeatedly reaches roughly 73--76 deg on the hard
        # ~2.3 m point-to-point steps without flipping.
        # ============================================================

        self.max_tilt_rad = math.radians(75.0)

        # Measured B3 static thrust law:
        #
        # F_motor(u) =
        #     0.187453678*u + 0.126414663*u^2
        #
        # At u=1:
        #     F_motor ~= 0.313868 N
        #     total   ~= 1.25547 N
        #
        # Use a very slightly conservative round value.
        self.max_collective_thrust_N = 1.25

        # Direct physical torque envelope.
        #
        # This is deliberately not a motor digital-twin parameter.
        # It prevents an ideal wrench actuator from producing absurd
        # moments unavailable to the real B3.
        self.max_torque_Nm = torch.tensor(
            [0.008, 0.008, 0.003],
            device=self.device,
            dtype=self.dtype,
        )

        # ============================================================
        # CLEAN INNER ATTITUDE LOOP
        #
        # Choose physical gains by specifying the desired closed-loop
        # rotational dynamics instead of importing PWM-space gains.
        #
        # For:
        #     J*qdd + Komega*qd + KR*q = 0
        #
        # choose:
        #     KR     = J * wn^2
        #     Komega = 2*zeta*J*wn
        #
        # wn=15 rad/s gives a fast but well-damped response on roll/
        # pitch and produces initial moments of the same few-mNm scale
        # implied by the real B3 transient.
        # ============================================================

        self.wn = torch.tensor(
            [15.0, 15.0, 10.0],
            device=self.device,
            dtype=self.dtype,
        )

        self.zeta = torch.tensor(
            [1.0, 1.0, 1.0],
            device=self.device,
            dtype=self.dtype,
        )

        self.KR = self.J * self.wn.square()

        self.Komega = (
            2.0
            * self.zeta
            * self.J
            * self.wn
        )

        self.reset()

        print()
        print("=" * 100)
        print("B3 BEHAVIOR GEOMETRIC CONTROLLER")
        print("=" * 100)
        print("outer loop       : real B3 Mellinger position law")
        print("inner loop       : clean SI-unit SO(3)")
        print("outer rate       :", self.outer_hz, "Hz")
        print("inner rate       :", self.inner_hz, "Hz")
        print("mass             :", float(self.mass), "kg")
        print("J                :", self.J.detach().cpu())
        print("kp               :", self.kp.detach().cpu())
        print("kd               :", self.kd.detach().cpu())
        print("ki               :", self.ki.detach().cpu())
        print(
            "max desired tilt :",
            math.degrees(self.max_tilt_rad),
            "deg",
        )
        print(
            "max thrust       :",
            self.max_collective_thrust_N,
            "N",
        )
        print(
            "max torque       :",
            self.max_torque_Nm.detach().cpu(),
            "Nm",
        )
        print("wn               :", self.wn.detach().cpu())
        print("KR               :", self.KR.detach().cpu())
        print("Komega           :", self.Komega.detach().cpu())
        print("=" * 100)

    def reset(self):
        self.step_count = 0

        self.int_pos_error = torch.zeros(
            3,
            device=self.device,
            dtype=self.dtype,
        )

        self.target_force_w = torch.tensor(
            [0.0, 0.0, 0.046 * 9.81],
            device=self.device,
            dtype=self.dtype,
        )

        self.R_des = torch.eye(
            3,
            device=self.device,
            dtype=self.dtype,
        )

    @staticmethod
    def _vee(M):
        return torch.stack(
            (
                M[2, 1],
                M[0, 2],
                M[1, 0],
            )
        )

    def _quat_wxyz_to_rotmat(self, q):
        q = q / torch.linalg.vector_norm(q)

        w, x, y, z = q

        return torch.stack(
            (
                torch.stack(
                    (
                        1.0 - 2.0 * (y*y + z*z),
                        2.0 * (x*y - w*z),
                        2.0 * (x*z + w*y),
                    )
                ),
                torch.stack(
                    (
                        2.0 * (x*y + w*z),
                        1.0 - 2.0 * (x*x + z*z),
                        2.0 * (y*z - w*x),
                    )
                ),
                torch.stack(
                    (
                        2.0 * (x*z - w*y),
                        2.0 * (y*z + w*x),
                        1.0 - 2.0 * (x*x + y*y),
                    )
                ),
            )
        )

    def _limit_target_force(self, F):
        """
        Enforce a real-B3-like force/tilt envelope without changing
        the position command itself.
        """
        F = F.clone()

        # The quadrotor must retain a positive desired body-z component.
        Fz = torch.clamp(
            F[2],
            min=torch.tensor(
                1.0e-3,
                device=self.device,
                dtype=self.dtype,
            ),
        )

        Fxy = F[:2]
        Fxy_norm = torch.linalg.vector_norm(Fxy)

        max_Fxy = (
            Fz
            * math.tan(self.max_tilt_rad)
        )

        if float(Fxy_norm) > float(max_Fxy):
            F[:2] = (
                Fxy
                * max_Fxy
                / torch.clamp(
                    Fxy_norm,
                    min=1.0e-9,
                )
            )

        # Limit force-vector magnitude to the real B3 collective
        # capability while preserving its desired direction.
        Fnorm = torch.linalg.vector_norm(F)

        if float(Fnorm) > self.max_collective_thrust_N:
            F = (
                F
                * self.max_collective_thrust_N
                / Fnorm
            )

        return F

    def _update_outer_loop(
        self,
        pos_w,
        vel_w,
        goal_pos_w,
        goal_yaw_rad,
    ):
        dt = 1.0 / self.outer_hz

        pos_error = goal_pos_w - pos_w
        vel_error = -vel_w

        self.int_pos_error = (
            self.int_pos_error
            + pos_error * dt
        )

        self.int_pos_error = torch.maximum(
            torch.minimum(
                self.int_pos_error,
                self.int_pos_limit,
            ),
            -self.int_pos_limit,
        )

        # Exact structure verified from Crazyflie Mellinger:
        #
        # target_thrust =
        #     m * (a_des - gravity)
        #     + kp * ep
        #     + kd * ev
        #     + ki * integral(ep)
        #
        # a_des = 0 for this static point-to-point benchmark.
        F = (
            -self.mass * self.gravity
            + self.kp * pos_error
            + self.kd * vel_error
            + self.ki * self.int_pos_error
        )

        F = self._limit_target_force(F)

        self.target_force_w = F

        b3_des = (
            F
            / torch.clamp(
                torch.linalg.vector_norm(F),
                min=1.0e-9,
            )
        )

        yaw = torch.as_tensor(
            goal_yaw_rad,
            device=self.device,
            dtype=self.dtype,
        )

        b1_yaw = torch.stack(
            (
                torch.cos(yaw),
                torch.sin(yaw),
                torch.zeros_like(yaw),
            )
        )

        b2_des = torch.linalg.cross(
            b3_des,
            b1_yaw,
        )

        b2_des = (
            b2_des
            / torch.clamp(
                torch.linalg.vector_norm(
                    b2_des
                ),
                min=1.0e-9,
            )
        )

        b1_des = torch.linalg.cross(
            b2_des,
            b3_des,
        )

        self.R_des = torch.stack(
            (
                b1_des,
                b2_des,
                b3_des,
            ),
            dim=1,
        )

    def step(
        self,
        pos_w,
        quat_wxyz,
        lin_vel_w,
        ang_vel_w,
        goal_pos_w,
        goal_yaw_rad=0.0,
    ):
        pos_w = pos_w.to(
            device=self.device,
            dtype=self.dtype,
        )

        quat_wxyz = quat_wxyz.to(
            device=self.device,
            dtype=self.dtype,
        )

        lin_vel_w = lin_vel_w.to(
            device=self.device,
            dtype=self.dtype,
        )

        ang_vel_w = ang_vel_w.to(
            device=self.device,
            dtype=self.dtype,
        )

        goal_pos_w = goal_pos_w.to(
            device=self.device,
            dtype=self.dtype,
        )

        R = self._quat_wxyz_to_rotmat(
            quat_wxyz
        )

        # Isaac root angular velocity is world-frame.
        # Convert to body-frame for the geometric attitude loop.
        omega_b = (
            R.transpose(0, 1)
            @ ang_vel_w
        )

        if (
            self.step_count == 0
            or self.step_count % self.outer_div == 0
        ):
            self._update_outer_loop(
                pos_w=pos_w,
                vel_w=lin_vel_w,
                goal_pos_w=goal_pos_w,
                goal_yaw_rad=goal_yaw_rad,
            )

        # ------------------------------------------------------------
        # Standard SO(3) geometric attitude error:
        #
        # eR =
        #   1/2 vee(
        #       R_des^T R - R^T R_des
        #   )
        # ------------------------------------------------------------

        E = (
            0.5
            * (
                self.R_des.transpose(0, 1) @ R
                - R.transpose(0, 1) @ self.R_des
            )
        )

        eR = self._vee(E)

        # Desired body angular velocity is zero for a static
        # position+yaw setpoint.
        eOmega = omega_b

        Jomega = self.J * omega_b

        gyro_term = torch.linalg.cross(
            omega_b,
            Jomega,
        )

        torque = (
            -self.KR * eR
            -self.Komega * eOmega
            + gyro_term
        )

        torque = torch.maximum(
            torch.minimum(
                torque,
                self.max_torque_Nm,
            ),
            -self.max_torque_Nm,
        )

        # Firmware-style collective thrust:
        # project desired world force onto current body z.
        b3_current_w = R[:, 2]

        thrust = torch.dot(
            self.target_force_w,
            b3_current_w,
        )

        thrust = torch.clamp(
            thrust,
            min=0.0,
            max=self.max_collective_thrust_N,
        )

        desired_tilt = torch.acos(
            torch.clamp(
                self.R_des[2, 2],
                -1.0,
                1.0,
            )
        )

        current_tilt = torch.acos(
            torch.clamp(
                R[2, 2],
                -1.0,
                1.0,
            )
        )

        wrench = torch.cat(
            (
                thrust.reshape(1),
                torque,
            )
        )

        self.step_count += 1

        return {
            "wrench_body": wrench,
            "thrust_N": thrust.detach().clone(),
            "torque_Nm": torque.detach().clone(),
            "target_force_w": (
                self.target_force_w
                .detach()
                .clone()
            ),
            "R_des": self.R_des.detach().clone(),
            "eR": eR.detach().clone(),
            "omega_b": omega_b.detach().clone(),
            "desired_tilt_rad": (
                desired_tilt.detach().clone()
            ),
            "current_tilt_rad": (
                current_tilt.detach().clone()
            ),
            "position_error_w": (
                goal_pos_w - pos_w
            ).detach().clone(),
        }
