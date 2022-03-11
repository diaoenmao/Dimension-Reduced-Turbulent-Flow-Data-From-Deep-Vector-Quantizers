# Dimension Reduced Turbulent Flow Data From Deep Vector Quantizers
This is an implementation of [A Physics-Informed Vector Quantized Autoencoder for Data Compression of Turbulent Flow](https://arxiv.org/abs/2201.03617)  
This is an implementation of [Dimension Reduced Turbulent Flow Data From Deep Vector Quantizers](https://arxiv.org/abs/2103.01074)

## Requirements
 - see requirements.txt

## Instruction
 - Global hyperparameters are configured in config.yml
 - Hyperparameters can be found at process_control() in utils.py 

## Examples
 - Train vqvae, compression scale 1, regularization parameter <img src="https://render.githubusercontent.com/render/math?math=\alpha=0.1,\gamma=0">
    ```ruby
    python train_vqvae.py --data_name Turb --model_name vqvae --control_name 1_exact-physcis_0.1-0
    ```

 - Test vqvae, compression scale 3, regularization parameter <img src="https://render.githubusercontent.com/render/math?math=\alpha=0.1,\gamma=0.0001">
    ```ruby
    python test_vqvae.py --data_name Turb --model_name vqvae --control_name 3_exact-physcis_0.1-0.0001
    ```
    
## Results
- Schematic of the VQ-AE architecture.

![vqae](/asset/vqae.png)

- Comparing original and reconstructed 3D (a) stationary isotropic, (b) decaying isotropic, and (c) Taylor-Green vortex turbulence compressed by VQ-AE.

![velocity](/asset/velocity.png)

- (a) with and (b) without regularizations for PDFs of normalized longitudinal (left), transverse (middle) components of velocity gradient tensor, and Turbulence Kinetic Energy spectra (right) of stationary isotropic turbulence flow.

![regularization](/asset/regularization.png)

## Acknowledgement
*Mohammadreza Momenifar  
Enmao Diao  
Vahid Tarokh  
Andrew D. Bragg*
