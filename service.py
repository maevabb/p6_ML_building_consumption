import bentoml
from pydantic import BaseModel, model_validator, ValidationError, Field, field_validator
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import OneHotEncoder
from enum import Enum
from typing_extensions import Self

# Encoder OneHotEncoder
with open("onehotencoder.pkl", "rb") as f:
    enc = pickle.load(f)

#Liste des valeurs valides pour PrimaryPropertyType
class PrimaryPropertyTypeEnum(str, Enum):
    Distribution_Center = "Distribution_Center"
    Hospital = "Hospital"
    Hotel = "Hotel"
    Laboratory = "Laboritory"
    Medical_Office = "Medical_Office"
    Mixed_Use = "Mixed_Use"
    Office = "Office"
    Restaurant = "Restaurant"
    Retail_Store = "Retail_Store"
    School = "School"
    Self_Storage = "Self_Storage"
    Supermarket = "Supermarket"
    University = "University"
    Warehouse = "Warehouse"
    Worship = "Worship"
    Other = "Other"

# Liste des valeurs valides pour EnergyUsed
class EnergyUsedEnum(str, Enum):
    electricity = "electricity"
    gas = "gas"
    both = "both"
    none = "none"

# Dictionnaire de conversion des valeurs de EnergyUsed
energy_mapping = {'none': 0, 'gas': 1, 'both': 2, 'electricity': 3 }  

# Liste des valeurs valides pour Neighborhood
class NeighborhoodEnum(str, Enum):
    DOWNTOWN = "DOWNTOWN"
    SOUTH = "SOUTH"
    NORTH = "NORTH"
    EAST = "EAST"

@bentoml.service()
class EnergyPrediction:
    
    model = bentoml.models.get("energy_consumption_model:latest")

    def __init__(self):
        self.regressor = bentoml.sklearn.load_model(self.model)
   
    # Modèle Pydantic pour la validation des données
    class BuildingData(BaseModel):
        PrimaryPropertyType: PrimaryPropertyTypeEnum
        BuildingAge: int = Field(ge=0)
        PropertyGFATotal: float = Field(gt=0)
        PropertyGFABuilding: float = Field(ge=0)
        NumberofFloors: int = Field(ge=0)
        EnergyUsed: EnergyUsedEnum
        HasSecondUse: bool
        Neighborhood: NeighborhoodEnum
        HasParking: bool
        NumberofBuildings: int = Field(ge=0)

        @model_validator(mode='after')
        def check_property_gfa(self) -> Self:
            if self.PropertyGFATotal < self.PropertyGFABuilding:
                raise ValueError('PropertyGFATotal doit être égal ou supérieur à PropertyGFABuilding')
            return self
        
        @field_validator("EnergyUsed", mode='after')
        def convert_energy_used(cls, value:str) -> int:
            return energy_mapping[value]
        
        @field_validator("NumberofFloors", mode='after')
        def convert_floors_number(cls, value:int) -> int:
            if value <= 1:
                return 1
            elif value > 5 and value <= 10:
                return 6
            elif value > 10:
                return 7
            return value
        
        @field_validator("NumberofBuildings", mode='after')
        def convert_building_number(cls, value:int) -> int:
            if value <= 1:
                return 1
            elif value > 5 and value <= 10:
                return 6
            elif value > 10:
                return 7
            return value

    @bentoml.api
    def predict(self, data: BuildingData):
        # Convertir en DataFrame
        input_data = pd.DataFrame([data.model_dump()])
        
        # Appliquer OneHotEncoder sur PrimaryPropertyType et Neighborhood
        categorical_features = input_data[["PrimaryPropertyType", "Neighborhood"]]
        encoded_categorical = enc.transform(categorical_features).toarray()

        # Convertir en DataFrame avec les bonnes colonnes
        encoded_df = pd.DataFrame(encoded_categorical, columns=enc.get_feature_names_out(["PrimaryPropertyType", "Neighborhood"]))

        # Supprimer les colonnes d'origine et ajouter les nouvelles
        input_data = input_data.drop(columns=["PrimaryPropertyType", "Neighborhood"])
        input_data = pd.concat([input_data, encoded_df], axis=1)

        # Convertir les valeurs binaires (HasSecondUse et HasParking) en 0/1
        input_data["HasSecondUse"] = input_data["HasSecondUse"].astype(int)
        input_data["HasParking"] = input_data["HasParking"].astype(int)

        # Faire la prédiction
        prediction = self.regressor.predict(input_data).tolist()

        return {"SiteEUI(kBtu/sf)": prediction}