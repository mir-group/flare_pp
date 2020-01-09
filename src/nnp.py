import numpy as np
import ace
import torch

class NNP(torch.nn.Module):
    def __init__(self, nos, layers, input_size, activation,
                 descriptor_calculator, descriptor_method, cutoff):
        super(NNP, self).__init__()

        self.descriptor_calculator = descriptor_calculator
        self.descriptor_method = descriptor_method
        self.cutoff = cutoff

        for n in range(nos):
            setattr(self, "spec" + str(n),
                    SpeciesNet(layers, input_size, activation))
    
    def predict_local_EF(self, local_environment):
        # Initialize energy/force tensor.
        ef_tens = torch.zeros(1 + 3 * local_environment.noa)

        # Calculate descriptor.
        getattr(self.descriptor_calculator,
                self.descriptor_method)(local_environment)
        descriptor = \
            torch.tensor(self.descriptor_calculator.descriptor_vals).double()
        descriptor.requires_grad = True
        desc_grad_torch = torch.from_numpy(self.descriptor_calculator
                                           .descriptor_force_dervs)

        # Forward pass.
        spec = local_environment.central_index
        local_energy = getattr(self, "spec" + str(spec)).forward(descriptor)

        # Compute partial forces.
        net_grad = \
            torch.autograd.grad(local_energy, descriptor, create_graph=True)[0]

        # Store energy and partial forces.
        ef_tens[0] = local_energy
        ef_tens[1:] = torch.mv(desc_grad_torch, net_grad)

        return ef_tens

    def predict_E(self, structure):
        energy = 0
        for count, spec in enumerate(structure.species):
            # Compute environment and its descriptor.
            environment = ace.LocalEnvironment(structure, count, self.cutoff)
            getattr(self.descriptor_calculator, self.descriptor_method)\
                (environment)
            descriptor = \
                torch.tensor(self.descriptor_calculator
                             .descriptor_vals).double()

            # Pass descriptor through the corresponding species network.
            energy += getattr(self, "spec" + str(spec)).forward(descriptor)

        return energy

    def predict_EF(self, structure):
        pass

    def predict_EFS(self, structure):
        pass

    def update(self, structure, labels):
        pass

class SpeciesNet(torch.nn.Module):
    def __init__(self, layers, input_size, activation):
        super(SpeciesNet, self).__init__()

        # Store linear layers of the network.
        self.layer_count = len(layers) + 1

        setattr(self, "lin0", torch.nn.Linear(input_size, layers[0]).double())

        for n in range(1, len(layers)):
            # Connect previous hidden layer to current hidden layer.
            setattr(self, "lin"+str(n),
                    torch.nn.Linear(layers[n-1], layers[n]).double())

        # Connect final hidden layer to the output.
        setattr(self, "lin"+str(len(layers)),
                torch.nn.Linear(layers[-1], 1).double())

        # Set the activation function.
        self.activation = activation

    def forward(self, descriptor):
        x = descriptor
        for n in range(self.layer_count - 1):
            lin = getattr(self, "lin" + str(n))
            x = self.activation(lin(x))
        return getattr(self, "lin" + str(self.layer_count - 1))(x)