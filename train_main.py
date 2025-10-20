import os
import torch
from src.data_loader import get_data_loaders
from src.model import create_model
from src.train import train_model
from src.evaluate import evaluate_model

def main():
    # Create directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    # Check device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Get data loaders
    print('Loading data...')
    train_loader, val_loader, test_loader = get_data_loaders(batch_size=32)
    
    print(f'Training samples: {len(train_loader.dataset)}')
    print(f'Validation samples: {len(val_loader.dataset)}')
    print(f'Test samples: {len(test_loader.dataset)}')
    
    # Create model
    print('Creating model...')
    model = create_model(device)
    print(model)
    
    # Train model
    print('Starting training...')
    trained_model = train_model(model, train_loader, val_loader, num_epochs=20)
    
    # Evaluate model
    print('Evaluating model...')
    accuracy, predictions, labels, probabilities = evaluate_model(trained_model, test_loader)
    
    print(f'\nFinal Test Accuracy: {accuracy * 100:.2f}%')

if __name__ == '__main__':
    main()