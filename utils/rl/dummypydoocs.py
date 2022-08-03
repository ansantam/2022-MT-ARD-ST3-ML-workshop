import time

import cheetah
import numpy as np

import utils.rl.ARESlatticeStage3v1_9 as lattice


class DummyMachine:

    auxiliary_channels = [
        "SINBAD.RF/LLRF.CONTROLLER/VS.AR.LI.RSB.G.1/AMPL.SAMPLE",       # Gun amplitude
        "SINBAD.RF/LLRF.CONTROLLER/VS.AR.LI.RSB.G.1/PHASE.SAMPLE",      # Gun phase
        "SINBAD.RF/LLRF.CONTROLLER/PROBE.AR.LI.RSB.G.1/POWER.SAMPLE",   # Gun power
        "SINBAD.MAGNETS/MAGNET.ML/ARLIMSOG1+-/FIELD.RBV",               # Solenoid field at center
        "SINBAD.MAGNETS/MAGNET.ML/ARLIMSOG1+-/CURRENT.RBV",             # Solenoid current
        "SINBAD.MAGNETS/MAGNET.ML/ARLIMSOG1+-/MOMENTUM.SP",             # Solenoid beam momentum setting
        "SINBAD.MAGNETS/MAGNET.ML/ARLIMCHG1/KICK.RBV",                  # HCor (ARLIMCHG1) kick
        "SINBAD.MAGNETS/MAGNET.ML/ARLIMCHG1/CURRENT.RBV",               # HCor (ARLIMCHG1) current
        "SINBAD.MAGNETS/MAGNET.ML/ARLIMCHG1/MOMENTUM.SP",               # HCor (ARLIMCHG1) beam momentum setting
        "SINBAD.MAGNETS/MAGNET.ML/ARLIMCVG1/KICK.RBV",                  # VCor (ARLIMCVG1) kick
        "SINBAD.MAGNETS/MAGNET.ML/ARLIMCVG1/CURRENT.RBV",               # VCor (ARLIMCVG1) current
        "SINBAD.MAGNETS/MAGNET.ML/ARLIMCVG1/MOMENTUM.SP",               # VCor (ARLIMCVG1) beam momentum setting
        "SINBAD.UTIL/INJ.LASER.MOTOR/MOTOR1.MBDEV1/POS",                # Laser attenuation
        "SINBAD.LASER/SINBADCPULASER1.SETTINGS/SINBAD_aperture_pos/SETTINGS.CURRENT",   # Laser aperture
        "SINBAD.DIAG/SCREEN.MOTOR/MOTOR4.MBDEV4/POS",                   # Collimator x position
        "SINBAD.DIAG/COLLIMATOR.ML/AR.LI.SLH.G.1/POS",                  # Collimator y position
        "SINBAD.MAGNETS/MAGNET.ML/ARLIMCHG2/KICK.RBV",                  # HCor (ARLIMCHG2) kick
        "SINBAD.MAGNETS/MAGNET.ML/ARLIMCHG2/CURRENT.RBV",               # HCor (ARLIMCHG2) current
        "SINBAD.MAGNETS/MAGNET.ML/ARLIMCHG2/MOMENTUM.SP",               # HCor (ARLIMCHG2) beam momentum setting
        "SINBAD.MAGNETS/MAGNET.ML/ARLIMCVG2/KICK.RBV",                  # VCor (ARLIMCVG2) kick
        "SINBAD.MAGNETS/MAGNET.ML/ARLIMCVG2/CURRENT.RBV",               # VCor (ARLIMCVG2) current
        "SINBAD.MAGNETS/MAGNET.ML/ARLIMCVG2/MOMENTUM.SP",               # VCor (ARLIMCVG2) beam momentum setting
        "SINBAD.DIAG/DARKC_MON/AR.LI.BCM.G.1/CHARGE.CALC",              # Dark current charge (?) -> Actually charge
        "SINBAD.MAGNETS/MAGNET.ML/ARLIMSOL1/CURRENT.RBV",             # TWS1 current
        "SINBAD.RF/LLRF.CONTROLLER/CTRL.AR.LI.RSB.L.1/SP.PHASE",        # TWS1 phase
        "SINBAD.RF/LLRF.CONTROLLER/FORWARD.AR.LI.RSB.L.1/POWER.SAMPLE", # TWS1 power
        "SINBAD.MAGNETS/MAGNET.ML/ARLIMCHG3/KICK.RBV",                  # HCor (ARLIMCHG3) kick
        "SINBAD.MAGNETS/MAGNET.ML/ARLIMCHG3/CURRENT.RBV",               # HCor (ARLIMCHG3) current
        "SINBAD.MAGNETS/MAGNET.ML/ARLIMCHG3/MOMENTUM.SP",               # HCor (ARLIMCHG3) beam momentum setting
        "SINBAD.MAGNETS/MAGNET.ML/ARLIMCVG3/KICK.RBV",                  # VCor (ARLIMCVG3) kick
        "SINBAD.MAGNETS/MAGNET.ML/ARLIMCVG3/CURRENT.RBV",               # VCor (ARLIMCVG3) current
        "SINBAD.MAGNETS/MAGNET.ML/ARLIMCVG3/MOMENTUM.SP",               # VCor (ARLIMCVG3) beam momentum setting
        "SINBAD.MAGNETS/MAGNET.ML/ARLIMCHM1/KICK.RBV",                  # HCor (ARLIMCHM1) kick
        "SINBAD.MAGNETS/MAGNET.ML/ARLIMCHM1/CURRENT.RBV",               # HCor (ARLIMCHM1) current
        "SINBAD.MAGNETS/MAGNET.ML/ARLIMCHM1/MOMENTUM.SP",               # HCor (ARLIMCHM1) beam momentum setting
        "SINBAD.MAGNETS/MAGNET.ML/ARLIMCVM1/KICK.RBV",                  # VCor (ARLIMCVM1) kick
        "SINBAD.MAGNETS/MAGNET.ML/ARLIMCVM1/CURRENT.RBV",               # VCor (ARLIMCVM1) current
        "SINBAD.MAGNETS/MAGNET.ML/ARLIMCVM1/MOMENTUM.SP",               # VCor (ARLIMCVM1) beam momentum setting
        "SINBAD.RF/LLRF.CONTROLLER/CTRL.AR.LI.RSB.L.2/SP.AMPL",         # TWS2 current
        "SINBAD.RF/LLRF.CONTROLLER/CTRL.AR.LI.RSB.L.2/SP.PHASE",        # TWS2 phase
        "SINBAD.RF/LLRF.CONTROLLER/FORWARD.AR.LI.RSB.L.2/POWER.SAMPLE", # TWS2 power
        "SINBAD.MAGNETS/MAGNET.ML/ARLIMCHG4/KICK.RBV",                  # HCor (ARLIMCHG4) kick
        "SINBAD.MAGNETS/MAGNET.ML/ARLIMCHG4/CURRENT.RBV",               # HCor (ARLIMCHG4) current
        "SINBAD.MAGNETS/MAGNET.ML/ARLIMCHG4/MOMENTUM.SP",               # HCor (ARLIMCHG4) beam momentum setting
        "SINBAD.MAGNETS/MAGNET.ML/ARLIMCVG4/KICK.RBV",                  # VCor (ARLIMCVG4) kick
        "SINBAD.MAGNETS/MAGNET.ML/ARLIMCVG4/CURRENT.RBV",               # VCor (ARLIMCVG4) current
        "SINBAD.MAGNETS/MAGNET.ML/ARLIMCVG4/MOMENTUM.SP",               # VCor (ARLIMCVG4) beam momentum setting
        "SINBAD.MAGNETS/MAGNET.ML/AREAMQZM1/STRENGTH.RBV",              # Q1 K1
        "SINBAD.MAGNETS/MAGNET.ML/AREAMQZM1/CURRENT.RBV",               # Q1 current
        "SINBAD.MAGNETS/MAGNET.ML/AREAMQZM1/MOMENTUM.SP",               # Q1 beam momentum setting
        "SINBAD.MAGNETS/MAGNET.ML/AREAMQZM2/STRENGTH.RBV",              # Q2 K1
        "SINBAD.MAGNETS/MAGNET.ML/AREAMQZM2/CURRENT.RBV",               # Q2 current
        "SINBAD.MAGNETS/MAGNET.ML/AREAMQZM2/MOMENTUM.SP",               # Q2 beam momentum setting
        "SINBAD.MAGNETS/MAGNET.ML/AREAMCVM1/KICK_MRAD.SP",             # CV kick (mrad)
        "SINBAD.MAGNETS/MAGNET.ML/AREAMCVM1/KICK_MRAD.RBV",             # CV kick (mrad)
        "SINBAD.MAGNETS/MAGNET.ML/AREAMCVM1/CURRENT.RBV",               # CV current
        "SINBAD.MAGNETS/MAGNET.ML/AREAMCVM1/MOMENTUM.SP",               # CV beam momentum setting
        "SINBAD.MAGNETS/MAGNET.ML/AREAMQZM3/STRENGTH.RBV",              # Q3 K1
        "SINBAD.MAGNETS/MAGNET.ML/AREAMQZM3/CURRENT.RBV",               # Q3 current
        "SINBAD.MAGNETS/MAGNET.ML/AREAMQZM3/MOMENTUM.SP",               # Q3 beam momentum setting
        "SINBAD.MAGNETS/MAGNET.ML/AREAMCHM1/KICK_MRAD.RBV",             # CH kick (mrad)
        "SINBAD.MAGNETS/MAGNET.ML/AREAMCHM1/CURRENT.RBV",               # CH current
        "SINBAD.MAGNETS/MAGNET.ML/AREAMCHM1/MOMENTUM.SP",               # CH beam momentum setting
        "SINBAD.DIAG/CAMERA/AR.EA.BSC.R.1/BINNINGHORIZONTAL",           # Screen horizontal binning 
        "SINBAD.DIAG/CAMERA/AR.EA.BSC.R.1/BINNINGVERTICAL",             # Screen vertical binning
        "SINBAD.DIAG/CAMERA/AR.EA.BSC.R.1/WIDTH",                       # Screen image width (pixel)
        "SINBAD.DIAG/CAMERA/AR.EA.BSC.R.1/HEIGHT",                      # Screen image height (pixel)
        "SINBAD.DIAG/CAMERA/AR.EA.BSC.R.1/GAINRAW",                     # Screen camera gain
        "SINBAD.DIAG/CAMERA/AR.EA.BSC.R.1/GAINAUTO",                    # Screen camera auto gain setting
        # "SINBAD.DIAG/CAMERA/AR.EA.BSC.R.1/IMAGE_EXT_ZMQ",               # Screen image
        "SINBAD.DIAG/CAMERA/AR.EA.BSC.R.1/SPECTRUM.X.MEAN",             # Beam mu x
        "SINBAD.DIAG/CAMERA/AR.EA.BSC.R.1/SPECTRUM.Y.MEAN",             # Beam mu y
        "SINBAD.DIAG/CAMERA/AR.EA.BSC.R.1/SPECTRUM.X.SIG",              # Beam sigma x
        "SINBAD.DIAG/CAMERA/AR.EA.BSC.R.1/SPECTRUM.Y.SIG",              # Beam sigma y
    ]

    def __init__(self):
        self.segment = cheetah.Segment.from_ocelot(lattice.cell, warnings=False).subcell("ARLIBSCR2", "AREABSCR1")
        self.segment.ARLIBSCR2.resolution = (2448, 2040)
        self.segment.ARLIBSCR2.pixel_size = (3.5488e-6, 2.5003e-6)
        self.segment.ARLIBSCR2.is_active = False
        self.segment.ARLIBSCR2.binning = 4
        
        
        
        self.segment.AREABSCR1.resolution = (2448, 2040)
        self.segment.AREABSCR1.pixel_size = (3.5488e-6, 2.5003e-6)
        self.segment.AREABSCR1.is_active = True

        self.segment.AREABSCR1.binning = 4

        self.beam_parameters = {
            "n": int(1e5),
            "mu_x": np.random.uniform(-3e-3, 3e-3),
            "mu_y": np.random.uniform(-3e-4, 3e-4),
            "mu_xp": np.random.uniform(-1e-4, 1e-4),
            "mu_yp": np.random.uniform(-1e-4, 1e-4),
            "sigma_x": np.random.uniform(0, 2e-3),
            "sigma_y": np.random.uniform(0, 2e-3),
            "sigma_xp": np.random.uniform(0, 1e-4),
            "sigma_yp": np.random.uniform(0, 1e-4),
            "sigma_s": np.random.uniform(0, 2e-3),
            "sigma_p": np.random.uniform(0, 5e-3),
            "energy": np.random.uniform(80e6, 160e6)
        }

        self.segment.AREAMQZM1.misalignment = (
            np.random.uniform(-200e-6, 200e-6),
            np.random.uniform(-200e-6, 200e-6)
        )
        self.segment.AREAMQZM2.misalignment = (
            np.random.uniform(-200e-6, 200e-6),
            np.random.uniform(-200e-6, 200e-6)
        )
        self.segment.AREAMQZM3.misalignment = (
            np.random.uniform(-200e-6, 200e-6),
            np.random.uniform(-200e-6, 200e-6)
        )
        self.segment.AREABSCR1.misalignment = (
            np.random.uniform(-200e-6, 200e-6),
            np.random.uniform(-200e-6, 200e-6)
        )

        self.areamqzm1_busy = False
        self.areamqzm2_busy = False
        self.areamqzm3_busy = False
        self.areamcvm1_busy = False
        self.areamchm1_busy = False

        self.areamqzm1_tw = time.time()
        self.areamqzm2_tw = time.time()
        self.areamqzm3_tw = time.time()
        self.areamcvm1_tw = time.time()
        self.areamchm1_tw = time.time()

        self.is_laser_on = True
    
    def write(self, channel, value):
        if channel == "SINBAD.DIAG/CAMERA/AR.EA.BSC.R.1/BINNINGHORIZONTAL":
            self.segment.AREABSCR1.binning = value
        elif channel == "SINBAD.DIAG/CAMERA/AR.EA.BSC.R.1/BINNINGVERTICAL":
            self.segment.AREABSCR1.binning = value
        elif channel == "SINBAD.MAGNETS/MAGNET.ML/AREAMQZM1/STRENGTH.SP":
            self.segment.AREAMQZM1.k1 = value
            self.areamqzm1_busy = True
            self.areamqzm1_tw = time.time()
        elif channel == "SINBAD.MAGNETS/MAGNET.ML/AREAMQZM2/STRENGTH.SP":
            self.segment.AREAMQZM2.k1 = value
            self.areamqzm2_busy = True
            self.areamqzm2_tw = time.time()
        elif channel == "SINBAD.MAGNETS/MAGNET.ML/AREAMQZM3/STRENGTH.SP":
            self.segment.AREAMQZM3.k1 = value
            self.areamqzm3_busy = True
            self.areamqzm3_tw = time.time()
        elif channel == "SINBAD.MAGNETS/MAGNET.ML/AREAMCVM1/KICK_MRAD.SP":
            self.segment.AREAMCVM1.angle = value / 1000
            self.areamcvm1_busy = True
            self.areamcvm1_tw = time.time()
        elif channel == "SINBAD.MAGNETS/MAGNET.ML/AREAMCHM1/KICK_MRAD.SP":
            self.segment.AREAMCHM1.angle = value / 1000
            self.areamchm1_busy = True
            self.areamchm1_tw = time.time()
        elif channel == "SINBAD.MAGNETS/MAGNET.ML/AREAMQZM1/CURRENT.SP":
            pass
        elif channel == "SINBAD.MAGNETS/MAGNET.ML/AREAMQZM2/CURRENT.SP":
            pass
        elif channel == "SINBAD.MAGNETS/MAGNET.ML/AREAMQZM3/CURRENT.SP":
            pass
        elif channel == "SINBAD.MAGNETS/MAGNET.ML/AREAMCVM1/CURRENT.SP":
            pass
        elif channel == "SINBAD.MAGNETS/MAGNET.ML/AREAMCHM1/CURRENT.SP":
            pass
        elif channel == "SINBAD.MAGNETS/MAGNET.ML/AREAMQZM1/PS_ON":
            pass
        elif channel == "SINBAD.MAGNETS/MAGNET.ML/AREAMQZM2/PS_ON":
            pass
        elif channel == "SINBAD.MAGNETS/MAGNET.ML/AREAMQZM3/PS_ON":
            pass
        elif channel == "SINBAD.MAGNETS/MAGNET.ML/AREAMCVM1/PS_ON":
            pass
        elif channel == "SINBAD.MAGNETS/MAGNET.ML/AREAMCHM1/PS_ON":
            pass
        elif channel == "SINBAD.MAGNETS/MAGNET.ML/AREAMQZM1/CYCLE":
            self.areamqzm1_busy = True
        elif channel == "SINBAD.MAGNETS/MAGNET.ML/AREAMQZM2/CYCLE":
            self.areamqzm2_busy = True
        elif channel == "SINBAD.MAGNETS/MAGNET.ML/AREAMQZM3/CYCLE":
            self.areamqzm3_busy = True
        elif channel == "SINBAD.MAGNETS/MAGNET.ML/AREAMCVM1/CYCLE":
            self.areamcvm1_busy = True
        elif channel == "SINBAD.MAGNETS/MAGNET.ML/AREAMCHM1/CYCLE":
            self.areamchm1_busy = True
        elif channel == "SINBAD.DIAG/TIMER.CENTRAL/MASTER/EVENT5":
            self.is_laser_on = bool(value[0])
        # 4 correctors in the li section
        elif channel == "SINBAD.MAGNETS/MAGNET.ML/ARLIMCHM1/KICK_MRAD.SP":
            self.segment.ARLIMCHM1.angle = value / 1000
            self.arlimchm1_busy = True
            self.arlimchm1_tw = time.time()
        elif channel == "SINBAD.MAGNETS/MAGNET.ML/ARLIMCVM1/KICK_MRAD.SP":
            self.segment.ARLIMCVM1.angle = value / 1000
            self.arlimcvm1_busy = True
            self.arlimcvm1_tw = time.time()
        elif channel == "SINBAD.MAGNETS/MAGNET.ML/ARLIMCHM2/KICK_MRAD.SP":
            self.segment.ARLIMCHM2.angle = value / 1000
            self.arlimchm2_busy = True
            self.arlimchm2_tw = time.time()
        elif channel == "SINBAD.MAGNETS/MAGNET.ML/ARLIMCVM2/KICK_MRAD.SP":
            self.segment.ARLIMCVM2.angle = value / 1000
            self.arlimcvm2_busy = True
            self.arlimcvm2_tw = time.time()
        else:
            raise NotImplementedError(f"Writing channel \"{channel}\" is not implemented")

        return True

    def read(self, channel):
        if channel == "SINBAD.DIAG/CAMERA/AR.EA.BSC.R.1/BINNINGHORIZONTAL":
            data = self.segment.AREABSCR1.binning
        elif channel == "SINBAD.DIAG/CAMERA/AR.EA.BSC.R.1/BINNINGVERTICAL":
            data = self.segment.AREABSCR1.binning
        elif channel == "SINBAD.MAGNETS/MAGNET.ML/AREAMQZM1/STRENGTH.RBV":
            data = self.segment.AREAMQZM1.k1
        elif channel == "SINBAD.MAGNETS/MAGNET.ML/AREAMQZM2/STRENGTH.RBV":
            data = self.segment.AREAMQZM2.k1
        elif channel == "SINBAD.MAGNETS/MAGNET.ML/AREAMQZM3/STRENGTH.RBV":
            data = self.segment.AREAMQZM3.k1
        elif channel == "SINBAD.MAGNETS/MAGNET.ML/AREAMCVM1/KICK_MRAD.RBV":
            data = self.segment.AREAMCVM1.angle * 1000
        elif channel == "SINBAD.MAGNETS/MAGNET.ML/AREAMCHM1/KICK_MRAD.RBV":
            data = self.segment.AREAMCHM1.angle * 1000
        elif channel == "SINBAD.MAGNETS/MAGNET.ML/AREAMQZM1/CURRENT.SP":
            data = self.segment.AREAMQZM1.k1                # Weird dummy data
        elif channel == "SINBAD.MAGNETS/MAGNET.ML/AREAMQZM2/CURRENT.SP":
            data = self.segment.AREAMQZM2.k1                # Weird dummy data
        elif channel == "SINBAD.MAGNETS/MAGNET.ML/AREAMQZM3/CURRENT.SP":
            data = self.segment.AREAMQZM3.k1                # Weird dummy data
        elif channel == "SINBAD.MAGNETS/MAGNET.ML/AREAMCVM1/CURRENT.SP":
            data = self.segment.AREAMCVM1.angle * 1000      # Weird dummy data
        elif channel == "SINBAD.MAGNETS/MAGNET.ML/AREAMCHM1/CURRENT.SP":
            data = self.segment.AREAMCHM1.angle * 1000      # Weird dummy data
        elif channel == "SINBAD.MAGNETS/MAGNET.ML/AREAMQZM1/BUSY":
            if time.time() - self.areamqzm1_tw > 2:
                self.areamqzm1_busy = False
            data = self.areamqzm1_busy
        elif channel == "SINBAD.MAGNETS/MAGNET.ML/AREAMQZM2/BUSY":
            if time.time() - self.areamqzm2_tw > 2:
                self.areamqzm2_busy = False
            data = self.areamqzm2_busy
        elif channel == "SINBAD.MAGNETS/MAGNET.ML/AREAMQZM3/BUSY":
            if time.time() - self.areamqzm3_tw > 2:
                self.areamqzm3_busy = False
            data = self.areamqzm3_busy
        elif channel == "SINBAD.MAGNETS/MAGNET.ML/AREAMCVM1/BUSY":
            if time.time() - self.areamcvm1_tw > 2:
                self.areamcvm1_busy = False
            data = self.areamcvm1_busy
        elif channel == "SINBAD.MAGNETS/MAGNET.ML/AREAMCHM1/BUSY":
            if time.time() - self.areamchm1_tw > 2:
                self.areamchm1_busy = False
            data = self.areamchm1_busy
        # 4 correcotrs in the li
        elif channel == "SINBAD.MAGNETS/MAGNET.ML/ARLIMCHM1/BUSY":
            if time.time() - self.arlimchm1_tw > 1:
                self.arlimchm1_busy = False
            data = self.arlimchm1_busy
        elif channel == "SINBAD.MAGNETS/MAGNET.ML/ARLIMCVM1/BUSY":
            if time.time() - self.arlimcvm1_tw > 1:
                self.arlimcvm1_busy = False
            data = self.arlimcvm1_busy
        elif channel == "SINBAD.MAGNETS/MAGNET.ML/ARLIMCHM2/BUSY":
            if time.time() - self.arlimchm2_tw > 1:
                self.arlimchm2_busy = False
            data = self.arlimchm2_busy
        elif channel == "SINBAD.MAGNETS/MAGNET.ML/ARLIMCVM2/BUSY":
            if time.time() - self.arlimcvm2_tw > 1:
                self.arlimcvm2_busy = False
            data = self.arlimcvm2_busy
        elif channel == "SINBAD.MAGNETS/MAGNET.ML/ARLIMCHM1/PS_ON":
            data = True
        elif channel == "SINBAD.MAGNETS/MAGNET.ML/ARLIMCVM1/PS_ON":
            data = True
        elif channel == "SINBAD.MAGNETS/MAGNET.ML/ARLIMCHM2/PS_ON":
            data = True
        elif channel == "SINBAD.MAGNETS/MAGNET.ML/ARLIMCVM2/PS_ON":
            data = True
        elif channel == "SINBAD.MAGNETS/MAGNET.ML/ARLIMCVM1/KICK_MRAD.RBV":
            data = self.segment.ARLIMCVM1.angle * 1000      # Weird dummy data
        elif channel == "SINBAD.MAGNETS/MAGNET.ML/ARLIMCHM1/KICK_MRAD.RBV":
            data = self.segment.ARLIMCHM1.angle * 1000      # Weird dummy data
        elif channel == "SINBAD.MAGNETS/MAGNET.ML/ARLIMCVM2/KICK_MRAD.RBV":
            data = self.segment.ARLIMCVM2.angle * 1000      # Weird dummy data
        elif channel == "SINBAD.MAGNETS/MAGNET.ML/ARLIMCHM2/KICK_MRAD.RBV":
            data = self.segment.ARLIMCHM2.angle * 1000      # Weird dummy data
        elif channel == "SINBAD.MAGNETS/MAGNET.ML/AREAMQZM1/PS_ON":
            data = True
        elif channel == "SINBAD.MAGNETS/MAGNET.ML/AREAMQZM2/PS_ON":
            data = True
        elif channel == "SINBAD.MAGNETS/MAGNET.ML/AREAMQZM3/PS_ON":
            data = True
            # data = False if time.time() - self.areamqzm3_tw < 230 else True
        elif channel == "SINBAD.MAGNETS/MAGNET.ML/AREAMCVM1/PS_ON":
            data = True
        elif channel == "SINBAD.MAGNETS/MAGNET.ML/AREAMCHM1/PS_ON":
            data = True
        elif channel == "SINBAD.DIAG/TIMER.CENTRAL/MASTER/EVENT5":
            data = [1 if self.is_laser_on else 0]
        elif channel == "SINBAD.DIAG/CAMERA/AR.EA.BSC.R.1/IMAGE_EXT_ZMQ":
            self.track()
            data = self.segment.AREABSCR1.reading.cpu().numpy().astype("uint16")
            data = np.flipud(data)
        elif channel == "SINBAD.DIAG/CAMERA/AR.EA.BSC.R.1/BINNINGHORIZONTAL":  # Screen horizontal binning 
            data = self.segment.AREABSCR1.binning
        elif channel == "SINBAD.DIAG/CAMERA/AR.EA.BSC.R.1/BINNINGVERTICAL":    # Screen vertical binning 
            data = self.segment.AREABSCR1.binning
        elif channel == "SINBAD.DIAG/CAMERA/AR.EA.BSC.R.1/WIDTH":              # Screen image width (pixel) 
            data = int(self.segment.AREABSCR1.resolution[0] / self.segment.AREABSCR1.binning)
        elif channel == "SINBAD.DIAG/CAMERA/AR.EA.BSC.R.1/HEIGHT":             # Screen image height (pixel)     
            data = int(self.segment.AREABSCR1.resolution[1] / self.segment.AREABSCR1.binning)
        elif channel == "SINBAD.DIAG/CAMERA/AR.EA.BSC.R.1/X.POLY_SCALE":
            data = np.array([0.0, 0.0, -self.segment.AREABSCR1.pixel_size[0], 0.0]) * 1000
        elif channel == "SINBAD.DIAG/CAMERA/AR.EA.BSC.R.1/Y.POLY_SCALE":
            data = np.array([0.0, 0.0, -self.segment.AREABSCR1.pixel_size[1], 0.0]) * 1000
        elif channel == "SINBAD.UTIL/MACHINE.STATE/ACCLXSISRV04._SVR/SVR.ERROR_COUNT":
            data = 0
            # data = 0
            # if not hasattr(self, "terr"):
            #     self.terr = time.time()
            # elif time.time() - self.terr < 120:
            #     data = 1
        elif channel in self.auxiliary_channels:
            data = None
        else:
            raise NotImplementedError(f"Reading channel \"{channel}\" is not implemented")
        
        return {"data": data}

    def track(self):
        if self.is_laser_on:
            incoming = cheetah.ParticleBeam.from_parameters(**self.beam_parameters)
            # incoming = cheetah.Beam.from_astra("environments/ARES_Linac.1351.010")
        else:
            incoming = cheetah.Beam.empty

        self.segment(incoming)


dummy = DummyMachine()


def write(channel, value):
    """Pretend to write the given value to the given channel."""
    return dummy.write(channel, value)


def read(channel):
    """Pretend to read given channel. May return random value if none is available."""
    return dummy.read(channel)
