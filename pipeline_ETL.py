import pandas as pd
import numpy as np
from typing import List, Dict, Any

class ConnectivityETL:
    """Pipeline ETL para procesar datos de cobertura móvil y accesos por tecnología"""
    
    def __init__(self):
        self.cobertura_movil = None
        self.accesos = None
        self.df_final = None
    
    def extract_data(self) -> None:
        """Extrae los datos de los archivos CSV"""
        print("Extrayendo datos...")
        self.cobertura_movil = pd.read_csv("Cobertura_movil.csv")
        self.accesos = pd.read_csv("Accesos_por_tecnologia.csv")
        print(f"Cobertura móvil: {self.cobertura_movil.shape}")
        print(f"Accesos: {self.accesos.shape}")
    
    def transform_cobertura_movil(self) -> pd.DataFrame:
        """Transforma los datos de cobertura móvil"""
        print("Transformando datos de cobertura móvil...")
        df = self.cobertura_movil.copy()
        
        # Eliminar columnas innecesarias
        cols_to_drop = ['COD DEPARTAMENTO', 'COD MUNICIPIO', 'CABECERA MUNICIPAL', 'COD CENTRO POBLADO']
        df.drop(columns=cols_to_drop, inplace=True)
        
        # Limpiar nombres de columnas
        df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
        
        # Corregir nombre de columna
        if 'cobertuta_4g' in df.columns:
            df.rename(columns={"cobertuta_4g": "cobertura_4g"}, inplace=True) 
        
        if 'año' in df.columns:
            df.rename(columns={"año": "ano"}, inplace=True)

        # Convertir columnas de texto a string
        text_cols = ['proveedor', 'departamento', 'municipio', 'centro_poblado']
        for col in text_cols:
            if col in df.columns:
                df[col] = df[col].astype('string')
        
        # Convertir columnas de tecnología a boolean
        tech_cols = ['cobertura_2g', 'cobertura_3g', 'cobertura_4g', 'cobertura_5g', 
                    'cobertura_lte', 'cobertura_hspa+,_hspa+dc']
        
        for col in tech_cols:
            if col in df.columns:
                df[col] = (df[col].astype(str)
                          .str.strip()
                          .str.upper()
                          .map({'S': True, 'N': False})
                          .astype('boolean'))
        
        # Feature engineering
        df['total_tecnologias'] = df[tech_cols].sum(axis=1)
        df['tiene_4g_o_mas'] = (
            df['cobertura_4g'] | 
            df['cobertura_lte'] | 
            df['cobertura_5g']
        )
        
        # Eliminar duplicados
        df.drop_duplicates(inplace=True)
        
        return df
    
    def transform_accesos(self) -> pd.DataFrame:
        """Transforma los datos de accesos por tecnología"""
        print("Transformando datos de accesos...")
        df = self.accesos.copy()
        
        # Eliminar columnas innecesarias
        cols_to_drop = ['COD_DEPARTAMENTO', 'COD_MUNICIPIO', 'SEGMENTO']
        df.drop(columns=cols_to_drop, inplace=True) 
        
        # Limpiar nombres de columnas
        df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_") 
        
        if 'año' in df.columns:
            df.rename(columns={"año": "ano"}, inplace=True)
        
        # Convertir columnas de texto a string
        text_cols = ['proveedor', 'departamento', 'municipio', 'tecnologia']
        for col in text_cols:
            if col in df.columns:
                df[col] = df[col].astype('string')
        
        # Convertir columnas numéricas
        numeric_cols = ['velocidad_bajada', 'velocidad_subida', 'no_de_accesos']
        for col in numeric_cols:
            if col in df.columns:
                if df[col].dtype == 'object' or df[col].dtype.name == 'string':
                    df[col] = df[col].astype(str).str.replace(',', '.', regex=False)
                df[col] = df[col].astype(float)
        
        # Feature engineering
        df['velocidad_total'] = df['velocidad_bajada'] + df['velocidad_subida']
        df['tipo_tecnologia'] = df['tecnologia'].apply(self._clasificar_tecnologia)
        
        return df
    
    def _clasificar_tecnologia(self, tec: str) -> str:
        """Clasifica el tipo de tecnología"""
        if pd.isna(tec):
            return 'OTRA'
        
        tec = str(tec).upper()
        
        if tec in ['ADSL', 'XDSL', 'DSL']:
            return 'COBRE'
        elif 'FIBRA' in tec:
            return 'FIBRA'
        elif 'SATELITAL' in tec:
            return 'SATELITAL'
        elif 'WIFI' in tec or 'INALAMBRICO' in tec:
            return 'INALÁMBRICA'
        elif 'CABLE' in tec:
            return 'CABLE'
        else:
            return 'OTRA'
    
    def merge_datasets(self, df_cobertura: pd.DataFrame, df_accesos: pd.DataFrame) -> pd.DataFrame:
        """Combina los datasets agrupados"""
        print("Combinando datasets...")
        
        # Agrupar cobertura móvil
        cobertura_grouped = df_cobertura.groupby(
            ['ano', 'trimestre', 'departamento', 'municipio', 'proveedor'], 
            as_index=False
        ).agg({
            'centro_poblado': lambda x: ', '.join(sorted(set(x.dropna()))),
            'cobertura_2g': 'any',
            'cobertura_3g': 'any',
            'cobertura_hspa+,_hspa+dc': 'any',
            'cobertura_4g': 'any',
            'cobertura_lte': 'any',
            'cobertura_5g': 'any',
            'total_tecnologias': 'mean',
            'tiene_4g_o_mas': 'any'
        })
        
        # Agrupar accesos
        accesos_grouped = df_accesos.groupby(
            ['ano', 'trimestre', 'departamento', 'municipio', 'proveedor'], 
            as_index=False
        ).agg({
            'tecnologia': lambda x: ', '.join(sorted(set(x.dropna()))),
            'velocidad_bajada': 'mean',
            'velocidad_subida': 'mean',
            'no_de_accesos': 'sum',
            'velocidad_total': 'mean',
            'tipo_tecnologia': lambda x: ', '.join(sorted(set(x.dropna())))
        })
        
        # Merge final
        df_final = pd.merge(
            cobertura_grouped, 
            accesos_grouped, 
            on=['ano', 'trimestre', 'departamento', 'municipio', 'proveedor'], 
            how='inner'
        )
        
        return df_final
    
    def load_data(self, df: pd.DataFrame, filename: str) -> None:
        """Carga los datos transformados a archivo CSV"""
        print(f"Cargando datos a {filename}...")
        df.to_csv(filename, index=False)
        print(f"Datos guardados exitosamente: {df.shape}")
    
    def run_pipeline(self) -> pd.DataFrame:
        """Ejecuta el pipeline completo"""
        print("=== INICIANDO PIPELINE ETL ===")
        
        # Extract
        self.extract_data()
        
        # Transform
        cobertura_transformed = self.transform_cobertura_movil()
        accesos_transformed = self.transform_accesos()
        
        # Load datos intermedios
        self.load_data(cobertura_transformed, "cobertura_movil_ETL.csv")
        self.load_data(accesos_transformed, "accesos_ETL.csv")
        
        # Merge final
        self.df_final = self.merge_datasets(cobertura_transformed, accesos_transformed)
        
        # Load resultado final
        self.load_data(self.df_final, "merge_ETL.csv")
        
        print("=== PIPELINE COMPLETADO ===")
        return self.df_final

# Uso del pipeline
if __name__ == "__main__":
    etl = ConnectivityETL()
    resultado = etl.run_pipeline()
    
    print("\nResumen del resultado final:")
    print(f"Forma del dataset: {resultado.shape}")
    print(f"Columnas: {list(resultado.columns)}")
    print(f"Primeras 5 filas:")
    print(resultado.head())
