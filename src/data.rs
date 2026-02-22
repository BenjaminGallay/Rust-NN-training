use burn::data::dataloader::batcher::Batcher;
use burn::data::dataset::Dataset;
use burn::prelude::*;
use rand::Rng;

#[derive(Clone)]
pub struct RandBatcher<B: Backend> {
    device: B::Device,
}

impl<B: Backend> RandBatcher<B> {
    pub fn new(device: B::Device) -> Self {
        Self { device }
    }
}

#[derive(Clone, Debug)]
pub struct RandBatch<B: Backend> {
    pub inputs: Tensor<B, 2>,
    pub specification_matrix: Tensor<B, 2>,
    pub specification_vector: Tensor<B, 1>,
}

impl<B: Backend> Batcher<B, Tensor<B, 1>, RandBatch<B>> for RandBatcher<B> {
    fn batch(
        &self,
        items: Vec<Tensor<B, 1>>,
        device: &<B as burn::prelude::Backend>::Device,
    ) -> RandBatch<B> {
        let inputs: Tensor<B, 2> = Tensor::stack(items, 0);

        let specification_matrix = Tensor::<B, 2>::from_data([[2]], &device); //Tensor::<B, 2>::from_data([[3.0, 4.9, 2.0], [2.0, 1.9, 3.0], [6.0, 1.5, 7.0]], &device);
        let mut arr = [0.0f32; 1];
        let mut rng = rand::rng();
        for i in 0..1 {
            arr[i] = rng.random_range(-0.1..0.1);
        }
        let specification_vector = Tensor::<B, 1>::from_data(arr, &device);

        RandBatch {
            inputs: inputs,
            specification_matrix,
            specification_vector,
        }
    }
}

#[derive(Clone, Debug)]
pub struct RandDataset<B: Backend> {
    pub vector: Vec<Tensor<B, 1>>,
    pub device: B::Device,
}

impl<B: Backend> RandDataset<B> {
    pub fn generate(device: B::Device, size: usize) -> Self {
        let mut vec = Vec::new();
        let mut rng = rand::rng();
        for _ in 0..size {
            let mut arr = [0.0f32; 1];
            for i in 0..1 {
                arr[i] = rng.random_range(-1.0..1.0);
            }
            let input_element = Tensor::<B, 1, Float>::from_data(arr, &device);
            vec.push(input_element);
        }
        RandDataset {
            vector: vec,
            device,
        }
    }
}

//Inutile mais demandé par l'implémentation de Burn

impl<B: Backend> Dataset<Tensor<B, 1>> for RandDataset<B> {
    fn get(&self, _index: usize) -> Option<burn::tensor::Tensor<B, 1>> {
        let mut arr = [0.0f32; 1];
        let mut rng = rand::rng();
        for i in 0..1 {
            arr[i] = rng.random_range(-1.0..1.0);
        }
        Some(Tensor::<B, 1, Float>::from_data(arr, &self.device))
    }
    fn len(&self) -> usize {
        self.vector.len()
    }
}
