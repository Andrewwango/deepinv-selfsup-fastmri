from deepinv.transform.base import Transform

class IdentityTransform(Transform):
     def _get_params(self, *args):
         return {}
     
     def _transform(self, x, **params):
         return x