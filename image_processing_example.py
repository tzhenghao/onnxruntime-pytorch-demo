# Third party imports
import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision.models import resnet18, ResNet18_Weights


class Predictor(nn.Module):

    def __init__(self):
        super().__init__()
        weights = ResNet18_Weights.DEFAULT
        self.resnet18 = resnet18(weights=weights, progress=False).eval()
        self.transforms = weights.transforms()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            x = self.transforms(x)
            y_pred = self.resnet18(x)
            return y_pred.argmax(dim=1)

if __name__ == "__main__":
    model_name = "bert-large-uncased-whole-word-masking-finetuned-squad"

    tokenizer = transformers.BertTokenizer.from_pretrained(model_name)
    model = transformers.BertForQuestionAnswering.from_pretrained(model_name)

    # Save the model parameters
    torch.save(model.state_dict(), PATH)

    # Redeclare the model and load the saved parameters
    model = TheModel(...)
    model.load_state_dict(torch.load(PATH))
    model.eval()
