"""Compute scores with Chemprop models from SyntheMol"""

__all__ = ["Chemprop_S_aureus", "Chemprop_solubility"]

from pathlib import Path

import numpy as np
import torch
from chemprop.models import MoleculeModel
from chemprop.utils import load_checkpoint, load_scalers
from rdkit.Chem import rdNormalizedDescriptors
from sklearn.preprocessing import StandardScaler
from typing import List, Union

from ..add_tag import add_tag
from ..component_results import ComponentResults
from ..normalize import normalize_smiles


def compute_rdkit_fingerprint(smiles: str) -> np.ndarray:
    """Generates RDKit 2D normalized features for a molecule.

    :param smiles: A SMILES string.
    :return: A 1D numpy array containing the RDKit 2D normalized features.
    """
    generator = rdNormalizedDescriptors.RDKit2DNormalized()
    rdkit_fp = generator.process(smiles)[1:]
    rdkit_fp = np.where(np.isnan(rdkit_fp), 0, rdkit_fp)
    rdkit_fp = rdkit_fp.astype(np.float32)

    return rdkit_fp


def chemprop_predict_on_molecule(
    model: MoleculeModel,
    smiles: str,
    fingerprint: np.ndarray,
    scaler: Union[StandardScaler, None] = None,
) -> float:
    """Predicts the property of a molecule using a Chemprop-RDKit model.

    :param model: A Chemprop model.
    :param smiles: A SMILES string.
    :param fingerprint: A 1D array of molecular fingerprints.
    :param scaler: A data scaler (if applicable).
    :return: The prediction on the molecule.
    """
    # Make prediction
    pred = model(batch=[[smiles]], features_batch=[fingerprint]).item()

    # Scale prediction if applicable
    if scaler is not None:
        pred = scaler.inverse_transform([[pred]])[0][0]

    return float(pred)


def chemprop_predict_on_molecule_ensemble(
    models: list[MoleculeModel],
    smiles: str,
    fingerprint: np.ndarray,
    scalers: List[StandardScaler],
) -> float:
    """Predicts the property of a molecule using an ensemble of Chemprop-RDKit models.

    :param models: An ensemble of Chemprop models.
    :param smiles: A SMILES string.
    :param fingerprint: A 1D array of molecular fingerprints.
    :param scalers: An ensemble of data scalers (if applicable).
    :return: The ensemble prediction on the molecule.
    """
    return float(
        np.mean(
            [
                chemprop_predict_on_molecule(model=model, smiles=smiles, fingerprint=fingerprint, scaler=scaler)
                for model, scaler in zip(models, scalers)
            ]
        )
    )


class ChempropScorer:
    """Scores molecules using a Chemprop-RDKit model or ensemble of models."""

    def __init__(
        self,
        model_path: Path,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        """Initialize the scorer.

        :param model_path: Path to a directory of model checkpoints (ensemble) or to a specific PT file.
        :param device: The device on which to run the model.
        """
        # Get model paths
        if model_path.is_dir():
            model_paths = list(model_path.glob("**/*.pt"))

            if len(model_paths) == 0:
                raise ValueError(f"Could not find any models in directory {model_path}.")
        else:
            model_paths = [model_path]

        # Load models
        self.models = [load_checkpoint(path=str(model_path), device=device).eval() for model_path in model_paths]

        # Load scalers
        self.scalers = [load_scalers(path=str(model_path))[0] for model_path in model_paths]

    def __call__(self, smiles: str) -> float:
        """Scores a molecule using a Chemprop-RDKit model or ensemble of models.

        :param smiles: A SMILES string.
        :return: The score of the molecule.
        """
        fingerprint = compute_rdkit_fingerprint(smiles=smiles)

        # Make prediction
        return chemprop_predict_on_molecule_ensemble(
            models=self.models,
            smiles=smiles,
            fingerprint=fingerprint,
            scalers=self.scalers,
        )


@add_tag("__component")
class Antibiotic:
    """Scores molecules for S. aureus antibiotic activity using a Chemprop-RDKit model from SyntheMol."""

    def __init__(self):
        self.scorer = ChempropScorer(
            model_path=Path("../../../SyntheMol/rl/models/s_aureus_chemprop_rdkit"),
        )

    @normalize_smiles
    def __call__(self, smiles: List[str]) -> np.ndarray:
        scores = [self.scorer(smiles) for smiles in smiles]
        return ComponentResults([np.array(scores)])


@add_tag("__component")
class Solubility:
    """Scores molecules for solubility using a Chemprop-RDKit model from SyntheMol."""

    def __init__(self):
        self.scorer = ChempropScorer(
            model_path=Path("../../../SyntheMol/rl/models/solubility_chemprop_rdkit"),
        )

    @normalize_smiles
    def __call__(self, smiles: List[str]) -> np.ndarray:
        scores = [self.scorer(smiles) for smiles in smiles]
        return ComponentResults([np.array(scores)])
