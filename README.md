## DL-Fit
Source code for the DL-Fit.


## Submitted Paper
The paper entitled: "A Joint 2.5D Physics-coupled Deep learning based Polynomial Fitting Approach for MR Electrical Properties Tomography" was submitted to IEEE Transactions on Medical Imaging (Kyu-Jin Jung & Thierry G. Meerbothe et al., 2024).


## Usage
* Util.py: Code for common utility function.

### DL-based phase-only polynomial fitting conductivity reconstruction 
* Util_Phase_Only.py: Utility code for phase-only polynomial fitting (for 2D).
* Util_Phase_Only_Joint.py: Utility code for phase-only polynomial fitting (for simultaneous training).
* 1_Initial_Training_Axial_Plane_Phase_Only.py, 1_Initial_Training_Coronal_Plane_Phase_Only.py, 1_Initial_Training_Sagittal_Plane_Phase_Only.py: Training code for each orthogonal plane.
* 2_Simultaneous_Training_Procedure_Phase_Only.py: Simultaneous training code for the physics-coupling procedure.
* 3_Testing_Procedure_Phase_Only.py: Testing code.

### DL-based polynomial fitting conductivity reconstruction with B1+ magnitude correction 
* Util_Mag_Corr.py: Utility code for polynomial fitting with B1+ magnitude correction (for 2D).
* Util_Mag_Corr_Joint.py: Utility code for polynomial fitting with B1+ magnitude correction (for simultaneous training).
* 1_Initial_Training_Axial_Plane_Mag_Corr.py, 1_Initial_Training_Coronal_Plane_Mag_Corr.py, 1_Initial_Training_Sagittal_Plane_Mag_Corr.py: Training code for each orthogonal plane.
* 2_Simultaneous_Training_Procedure_Mag_Corr.py: Simultaneous training code for the physics-coupling procedure.
* 3_Testing_Procedure_Mag_Corr.py: Testing code.


## Reference
Kyu-Jin Jung, Thierry G. Meerbothe,  Chuanjiang Cui, Mina Park, Cornelis A.T. van den Berg, Dong-Hyun Kim, and Stefano Mandija. A Joint 2.5D Physics-coupled Deep learning based Polynomial Fitting Approach for MR Electrical Properties Tomography. 2024 ISMRM & ISMRT Annual Meeting & Exhibition, Singapore (2024).

Kyu-Jin Jung, Thierry G. Meerbothe,  Chuanjiang Cui, Mina Park, Jaeuk Yi, Cornelis A.T. van den Berg, Dong-Hyun Kim, and Stefano Mandija. A Deep learning informed Polynomial Fitting Approach for Electrical Properties Tomography. 2023 ISMRM & ISMRT Annual Meeting & Exhibition, Toronto, Canada (2023).

## Acknowledgements
First authors: Kyu-Jin Jung and Thierry G. Meerbothe; ** Corresponding last authors: Stefano Mandija and Dong-Hyun Kim
