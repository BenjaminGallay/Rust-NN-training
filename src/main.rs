mod data;
mod inference;
mod model;
mod training;

use crate::model::ModelConfig;
use crate::training::TrainingConfig;

use burn::backend::{Autodiff, Wgpu};
use burn::optim::AdamConfig;

fn main() {
    type Backend = Wgpu<f32, i32>;
    type MyAutodiffBackend = Autodiff<Backend>;
    let device = burn::backend::wgpu::WgpuDevice::default();
    let artifact_dir = "./tmp/guide";

    crate::training::train::<MyAutodiffBackend>(
        artifact_dir,
        TrainingConfig::new(ModelConfig::new(10, 512), AdamConfig::new()),
        device.clone(),
    );

    crate::inference::infer::<Backend>(artifact_dir, device.clone(), 1000);

    // Matrice 4x3 avec laquelle on multiplie le vecteur de sortie (qui est de dimension 3x1)

    /*
    println!("Model input:\n{tensor:?}");

    let output = model.forward(tensor);

    println!("Model output:\n{output:?}");

    let output_prime = output.matmul(tensor_matrix.transpose());

    println!("Model output prime:\n{output_prime:?}");*/

    // On doit mtn prendre une matrice M 10x10 et faire M x output et regarder si c'est proche de input
    // Générer plein de input aléatoirement et comparer si M x output est proche de input et entraîner là-dessus
    // normalement pas de problème d'overfitting puisque tout est random
    // ensuite faire la vérification -_-
}
