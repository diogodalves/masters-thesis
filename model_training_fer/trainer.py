import os
import numpy as np
import copy

import torch
from sklearn.metrics import accuracy_score, \
                            precision_score, \
                            recall_score, \
                            f1_score, \
                            classification_report, \
                            confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

import torch.nn as nn
import torch.nn.utils.prune as prune

from tqdm import tqdm

def train_transformer_model(model,
                train_loader,
                val_loader,
                criterion,
                optimizer,
                *,
                model_name=None,
                device="cpu",
                epochs=100,
                patience=3):

    if not os.path.exists('model'):
        os.makedirs('model')

    global best_val_loss, patience_counter
    best_val_loss = np.inf
    patience_counter = 0
    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0

        for images, labels in tqdm(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images).logits
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            # Clean up GPU memory
            del images, labels, outputs, loss
            torch.cuda.empty_cache()

        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        avg_val_loss = evaluate_transformer_loss(model,
                                     val_loader,
                                     criterion,
                                     device)
        val_losses.append(avg_val_loss)

        print(f"Epoch {epoch + 1}, Train Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}")

        # Check for early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_model = model
            if model_name:
                torch.save(model.state_dict(), 'model/' + model_name)
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print("Early stopping triggered")
            break

    return best_model, train_losses, val_losses


def train_model(model,
                train_loader,
                val_loader,
                criterion,
                optimizer,
                *,
                model_name="model.pt",
                device="cpu",
                epochs=100,
                patience=3):

    if not os.path.exists('model'):
        os.makedirs('model')

    global best_val_loss, patience_counter
    best_val_loss = np.inf
    patience_counter = 0
    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0

        for images, labels in tqdm(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            # Clean up GPU memory
            del images, labels, outputs, loss
            torch.cuda.empty_cache()

        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        avg_val_loss = evaluate_loss(model,
                                     val_loader,
                                     criterion,
                                     device)
        val_losses.append(avg_val_loss)

        print(f"Epoch {epoch + 1}, Train Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}")

        # Check for early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            #torch.save(model.state_dict(), 'model/' + model_name)
            best_model = model
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print("Early stopping triggered")
            break

    return best_model, train_losses, val_losses

def play_the_lottery(model,
                    train_loader,
                    val_loader,
                    criterion,
                    optimizer,
                    epochs,
                    *,
                    device="cpu",
                    prune_percent=0.2,
                    total_prune_cycles=100,
                    patience=1,
                    cycle_patience=3):

    if not os.path.exists('model'):
        os.makedirs('model')

    best_model_wts = copy.deepcopy(model.state_dict())
    global_best_val_loss = np.inf  # Track the best validation loss across all cycles
    best_model_name = None
    train_losses = []
    val_losses = []
    previous_best_model_path = None  # Track the previous best model file

    # Calculate initial non-zero parameters for reference
    initial_params = count_nonzero_parameters(model)
    print(f"Initial non-zero parameters: {initial_params}")

    no_improvement_cycles = 0  # Track cycles with no improvement in validation loss

    for cycle in range(total_prune_cycles):
        print(f"Pruning cycle {cycle + 1}/{total_prune_cycles}")

        # Restore best weights and apply pruning masks before starting a new cycle
        if cycle > 0:
            model.load_state_dict(best_model_wts)
            restore_pruned_weights(model)

        current_params = count_nonzero_parameters(model)
        print(f"Params before cycle {cycle + 1}: {current_params}")

        cycle_best_val_loss = np.inf  # Best validation loss for the current cycle
        cycle_best_model_wts = copy.deepcopy(model.state_dict())  # Best model weights for the current cycle
        no_improvement_epochs = 0  # Track epochs with no improvement

        for epoch in range(epochs + cycle):
            model.train()
            total_train_loss = 0

            for images, labels in tqdm(train_loader, desc=f"Cycle {cycle + 1}, Epoch {epoch + 1}"):
                images = images.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                total_train_loss += loss.item()

                del images, labels, outputs, loss
                torch.cuda.empty_cache()

            avg_train_loss = total_train_loss / len(train_loader)
            train_losses.append(avg_train_loss)

            model.eval()
            total_val_loss = 0
            with torch.no_grad():
                for images, labels in val_loader:
                    images = images.to(device)
                    labels = labels.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    total_val_loss += loss.item()

                    del images, labels, outputs, loss
                    torch.cuda.empty_cache()

            avg_val_loss = total_val_loss / len(val_loader)
            val_losses.append(avg_val_loss)

            print(f"Epoch {epoch + 1}, Train Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}")

            # Check if the current epoch's validation loss is better than the cycle's best so far
            if avg_val_loss < cycle_best_val_loss:
                cycle_best_val_loss = avg_val_loss
                cycle_best_model_wts = copy.deepcopy(model.state_dict())
                no_improvement_epochs = 0  # Reset no improvement counter
            else:
                no_improvement_epochs += 1

            # Update global best model if necessary
            if avg_val_loss < global_best_val_loss:
                global_best_val_loss = avg_val_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                model_path = f'model/lottery_ticket_best_model_{cycle+1}_epoch_{epoch+1}.pt'
                torch.save(model.state_dict(), model_path)
                best_model_name = model_path

                # Remove previous best model file if it exists
                if previous_best_model_path is not None and os.path.exists(previous_best_model_path):
                    os.remove(previous_best_model_path)

                previous_best_model_path = model_path

            # Check for early stopping within the cycle
            if no_improvement_epochs >= patience:
                print(f"Early stopping triggered at epoch {epoch + 1} of cycle {cycle + 1}")
                break

        # Check for improvement between cycles
        print(f"Cycle best loss: {cycle_best_val_loss}")
        print(f"Global best loss: {global_best_val_loss}")
        if cycle_best_val_loss == global_best_val_loss:
            global_best_val_loss = cycle_best_val_loss
            no_improvement_cycles = 0  # Reset no improvement cycles counter
            best_model_wts = copy.deepcopy(cycle_best_model_wts)  # Ensure best model weights are retained
            print(f"New global best validation loss: {global_best_val_loss:.4f} at cycle {cycle + 1}")
        else:
            no_improvement_cycles += 1
            print(f"No improvement in cycle {cycle + 1}. Cycle patience: {no_improvement_cycles}/{cycle_patience}")

        if no_improvement_cycles >= cycle_patience:
            print(f"Early stopping after {cycle + 1} cycles with no improvement.")
            break

        try:
            prune_model(model, prune_percent)
        except:
            print("Achieved max amount for pruning")
            best_model_wts = copy.deepcopy(model.state_dict())

            current_params = count_nonzero_parameters(model)
            print(f"Parameters after cycle {cycle + 1}: {current_params}")
            break

        best_model_wts = copy.deepcopy(model.state_dict())

        # Count and print parameters to monitor pruning effectiveness
        current_params = count_nonzero_parameters(model)
        print(f"Parameters after cycle {cycle + 1}: {current_params}")

    # Load the best-pruned model and apply the final pruning masks
    model.load_state_dict(best_model_wts)

    restore_pruned_weights(model)

    # Calculate final non-zero parameters
    final_params = count_nonzero_parameters(model)
    print(f"Final non-zero parameters after all pruning cycles: {final_params}")

    return model, train_losses, val_losses, best_model_name

def prune_model(model, amount):
    total_params_before = 0
    total_params_after = 0
    for name, module in model.named_modules():
        if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
            if hasattr(module.weight, 'data'):
                original_non_zero = torch.sum(module.weight != 0).item()
                total_params_before += original_non_zero
                prune.l1_unstructured(module, name='weight', amount=amount)
                pruned_non_zero = torch.sum(module.weight != 0).item()
                total_params_after += pruned_non_zero

def restore_pruned_weights(model):
    for name, module in model.named_modules():
        if hasattr(module, '_mask'):
            module.weight.data = module.weight_orig * module._mask

def remove_pruning(model):
    for name, module in model.named_modules():
        if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
            if hasattr(module, 'weight'):
                if prune.is_pruned(module):
                    prune.remove(module, 'weight')

def store_pruning_masks(model):
    """
    Store the current pruning masks for all pruned layers in the model.
    
    Args:
    model (nn.Module): The model from which to store pruning masks.
    
    Returns:
    dict: A dictionary containing the pruning masks for each pruned layer.
    """
    pruning_masks = {}

    for name, module in model.named_modules():
        if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
            if prune.is_pruned(module):
                pruning_masks[name] = module.weight_mask.clone()

    return pruning_masks


def reapply_pruning_masks(model, pruning_masks):
    """
    Reapply the previously stored pruning masks to the corresponding layers.
    
    Args:
    model (nn.Module): The model to which the pruning masks will be reapplied.
    pruning_masks (dict): The stored pruning masks for each pruned layer.
    """
    for name, module in model.named_modules():
        if name in pruning_masks:
            prune.custom_from_mask(module, name='weight', mask=pruning_masks[name])



def count_nonzero_parameters(model):
    total_nonzero = sum(torch.sum(module.weight != 0).item() for name, module in model.named_modules() if hasattr(module, 'weight'))
    return total_nonzero


def apply_distillation(student_model,
                        teacher_model,
                        train_loader,
                        val_loader,
                        optimizer,
                        *,
                        model_name="model.pt",
                        num_epochs=20,
                        temperature=2.0,
                        alpha=0.5,
                        patience = 5,
                        device='cpu'):

    if not os.path.exists('model'):
        os.makedirs('model')

    best_val_loss = float('inf')
    patience_counter = 0

    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        student_model.train()
        running_loss = 0.0
        for images, labels in tqdm(train_loader):
            images, labels = images.to(device), labels.to(device)  # Move data to device
            optimizer.zero_grad()
            student_outputs = student_model(images)
            with torch.no_grad():
                teacher_outputs = teacher_model(images).logits
            loss = distillation_loss(student_outputs, teacher_outputs, temperature, alpha, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            del images, labels, student_outputs, loss
            torch.cuda.empty_cache()

        # Calculate average training loss for the epoch
        train_loss = running_loss / len(train_loader)
        train_losses.append(train_loss)

        # Validation loss
        student_model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)  # Move data to device
                student_outputs = student_model(images)
                loss = nn.CrossEntropyLoss()(student_outputs, labels)
                val_loss += loss.item()

                del images, labels, student_outputs, loss
                torch.cuda.empty_cache()

        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

        # Check early stopping criteria
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            #torch.save(student_model.state_dict(), 'model/' + model_name)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered")
                break

    return train_losses, val_losses

def distillation_loss(student_outputs, teacher_outputs, temperature, alpha, true_labels):
    soft_teacher_probs = torch.nn.functional.softmax(teacher_outputs / temperature, dim=1)
    soft_student_probs = torch.nn.functional.softmax(student_outputs / temperature, dim=1)
    distillation_loss = nn.KLDivLoss()(torch.log(soft_student_probs), soft_teacher_probs)
    student_loss = nn.CrossEntropyLoss()(student_outputs, true_labels)
    return alpha * distillation_loss + (1 - alpha) * student_loss

def evaluate_transformer_loss(model,
                  data_loader,
                  criterion,
                  device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images).logits
            loss = criterion(outputs, labels)
            total_loss += loss.item()
    avg_loss = total_loss / len(data_loader)
    return avg_loss

def evaluate_loss(model,
                  data_loader,
                  criterion,
                  device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
    avg_loss = total_loss / len(data_loader)
    return avg_loss

def plot_losses(train_losses, val_losses, title="Losses"):
    plt.figure(figsize=(10, 7))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title(title)
    plt.savefig(f'results/{title.lower().replace(" ", "_")}.png')
    plt.show()

def evaluate_transformer_model(model, test_loader, device, data, label_encoder=None):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images).logits
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='macro')
    recall = recall_score(all_labels, all_preds, average='macro')
    f1 = f1_score(all_labels, all_preds, average='macro')

    # Classification Report
    if label_encoder is not None:
        target_names = label_encoder.classes_
        class_report = classification_report(all_labels, all_preds, target_names=target_names)
    else:
        class_report = classification_report(all_labels, all_preds)

    # Plot Confusion Matrix
    conf_matrix = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 7))
    if label_encoder is not None:
        sns.heatmap(conf_matrix,
                    annot=True,
                    fmt='d',
                    cmap='Blues',
                    xticklabels=label_encoder.classes_,
                    yticklabels=label_encoder.classes_)
    else:
        sns.heatmap(conf_matrix,
                    annot=True, fmt='d',
                    cmap='Blues')

    plt.title(f'{data.upper()} Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(f'results/{data}_confusion_matrix.png')
    plt.show()

    # Print Metrics
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("Classification Report:\n", class_report)

def evaluate_model(model, test_loader, device, data, label_encoder=None):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='macro')
    recall = recall_score(all_labels, all_preds, average='macro')
    f1 = f1_score(all_labels, all_preds, average='macro')

    # Classification Report
    if label_encoder is not None:
        target_names = label_encoder.classes_
        class_report = classification_report(all_labels, all_preds, target_names=target_names)
    else:
        class_report = classification_report(all_labels, all_preds)

    # Plot Confusion Matrix
    conf_matrix = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 7))
    if label_encoder is not None:
        sns.heatmap(conf_matrix,
                    annot=True,
                    fmt='d',
                    cmap='Blues',
                    xticklabels=label_encoder.classes_,
                    yticklabels=label_encoder.classes_)
    else:
        sns.heatmap(conf_matrix,
                    annot=True, fmt='d',
                    cmap='Blues')

    plt.title(f'{data.upper()} Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(f'results/{data}_confusion_matrix.png')
    plt.show()

    # Print Metrics
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("Classification Report:\n", class_report)