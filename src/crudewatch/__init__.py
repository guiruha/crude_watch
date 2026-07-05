"""CrudeWatch: WTI crude futures dataset preparation and plotting.

Subpackages:
    crudewatch.infra             constants and IO helpers
    crudewatch.data_preparation  cleaning/expiry helpers and dataframe builders
    crudewatch.plots             the black & green chart theme (needs plotly)

``crudewatch.plots`` is intentionally NOT imported here so that importing the
package does not require the optional ``plotly`` dependency.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
