
from abc import ABC, abstractmethod
import pandas as pd

class GeneDS(ABC):
    def __init__(self,species,**kwargs):
        super().__init__()
        self.species = species
        self.client = self.get_client()
    
    @abstractmethod        
    def parse_key(self,key):
        pass
    
    @abstractmethod
    def get_client(self):
        pass

    @abstractmethod
    def query_genes(self,genes,return_info):
        pass
    
    
class MyGeneInfoDS(GeneDS):
    def __init__(self,key,species,**kwargs):
        super().__init__(species=species)
        self.key = self.parse_key(key)
        
    def parse_key(self, key):
        mapping = {'ensembl':'ensembl.gene'}
        return mapping[key]

    def get_client(self):
        
        from mygene import MyGeneInfo
        # GeneInfo client
        client = MyGeneInfo()
        return client

    def query_genes(self,genes,return_info):
        # Query multiple genes and request GO terms + summary + pathways
        annotations = self.client.querymany(
            genes,
            scopes=self.key,
            fields=return_info,
            species=self.species
        )
        return annotations
    
    def extract_info(self, field,value):
        if field=="pathway":
            hits = set()
            for source in value.values():
                if isinstance(source, dict):  # e.g. WikiPathways single dict
                    hits.add(source.get("name", ""))
                elif isinstance(source, list):
                    for entry in source:
                        if isinstance(entry, dict):
                            hits.add(entry.get("name", ""))
            return hits
        else:
            return value
    
    def parse_results(self, results):
        # Build structured summary table
        rows = []
        for result in results:
            id = result.get("query")
            row = {self.key: id}
            for field in result.keys():
                info = self.extract_info(field,result.get(field, ""))
                row[field] = info
            rows.append(row)
        result_df = pd.DataFrame(rows)
        return result_df
    

def get_gene_info(type,**kwargs):
    gene_info_classes = {
        'my_gene_info':MyGeneInfoDS
    }
    
    gene_info_instance = gene_info_classes[type](**kwargs)
    return gene_info_instance


