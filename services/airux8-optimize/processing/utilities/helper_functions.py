"""
Helper functions for data analysis and processing
"""

import pandas as pd
from typing import List, Dict, Optional


def analyze_feature_correlations(
    area_df: pd.DataFrame,
    target_features: List[str] = None,
    correlation_targets: List[str] = None
) -> Dict[str, pd.Series]:
    """
    Analyze correlations between features and target variables
    
    Args:
        area_df: DataFrame containing the data to analyze
        target_features: List of feature columns to analyze (optional)
        correlation_targets: List of target columns to correlate against (optional)
        
    Returns:
        Dictionary containing correlation results for each target
    """
    if target_features is None:
        # Import from config to ensure consistency
        from config.config_train import BASE_FEATURES
        target_features = BASE_FEATURES
    
    if correlation_targets is None:
        correlation_targets = ["Indoor Temp.", "adjusted_power"]
    
    # Find available features and targets
    available_feats = [col for col in target_features if col in area_df.columns]
    available_targets = [col for col in correlation_targets if col in area_df.columns]
    
    print(f"\n✅ 利用可能な特徴量 ({len(available_feats)}個):")
    for feat in available_feats:
        print(f"  - {feat}")
    
    missing_feats = [col for col in target_features if col not in area_df.columns]
    if missing_feats:
        print(f"\n⚠️ 不足している特徴量 ({len(missing_feats)}個):")
        for feat in missing_feats:
            print(f"  - {feat}")
    
    # Calculate correlations
    correlation_results = {}
    
    if available_feats and available_targets:
        # Create correlation matrix
        target_cols = available_feats + available_targets
        print(f"\n🔍 特徴量間の相関確認:")
        corr_matrix = area_df[target_cols].corr(numeric_only=True)
        
        # Analyze correlations for each target
        for target in available_targets:
            if target in corr_matrix.columns:
                # Calculate correlations with target
                target_corr = (
                    corr_matrix[target]
                    .drop(labels=[target] if target in corr_matrix.index else [])
                    .abs()
                    .sort_values(ascending=False)
                )
                
                correlation_results[target] = target_corr
                
                # Print results
                if target == "Indoor Temp.":
                    print(f"\n🌡️ 室温との相関 (上位10位):")
                    for feat, corr in target_corr.head(10).items():
                        print(f"  {feat}: {corr:.3f}")
                elif target == "adjusted_power":
                    print(f"\n⚡ 電力との相関 (上位10位):")
                    for feat, corr in target_corr.head(10).items():
                        print(f"  {feat}: {corr:.3f}")
                else:
                    print(f"\n📊 {target}との相関 (上位10位):")
                    for feat, corr in target_corr.head(10).items():
                        print(f"  {feat}: {corr:.3f}")
    
    return correlation_results


def print_correlation_summary(correlation_results: Dict[str, pd.Series]) -> None:
    """
    Print a summary of correlation results
    
    Args:
        correlation_results: Dictionary containing correlation results from analyze_feature_correlations
    """
    if not correlation_results:
        print("No correlation results to display")
        return
    
    print(f"\n{'='*60}")
    print("📊 Correlation Analysis Summary")
    print(f"{'='*60}")
    
    for target, correlations in correlation_results.items():
        print(f"\n🎯 Target: {target}")
        print(f"{'Feature':<25} | {'Correlation':<12}")
        print(f"{'-'*25}-+-{'-'*12}")
        
        # Show top 5 correlations
        for feature, corr in correlations.head(5).items():
            print(f"{feature:<25} | {corr:<12.3f}")
        
        if len(correlations) > 5:
            print(f"... and {len(correlations) - 5} more features")
    
    print(f"{'='*60}")


def get_top_correlated_features(
    correlation_results: Dict[str, pd.Series], 
    target: str, 
    top_n: int = 5
) -> List[str]:
    """
    Get top N most correlated features for a specific target
    
    Args:
        correlation_results: Dictionary containing correlation results
        target: Target variable name
        top_n: Number of top features to return
        
    Returns:
        List of top N feature names
    """
    if target not in correlation_results:
        return []
    
    return correlation_results[target].head(top_n).index.tolist()


def validate_data_quality(area_df: pd.DataFrame) -> Dict[str, any]:
    """
    Validate data quality and return summary statistics
    
    Args:
        area_df: DataFrame to validate
        
    Returns:
        Dictionary containing validation results
    """
    validation_results = {
        "total_rows": len(area_df),
        "total_columns": len(area_df.columns),
        "missing_values": {},
        "data_types": {},
        "zones": []
    }
    
    # Check for missing values
    for col in area_df.columns:
        missing_count = area_df[col].isnull().sum()
        if missing_count > 0:
            validation_results["missing_values"][col] = missing_count
    
    # Check data types
    for col in area_df.columns:
        validation_results["data_types"][col] = str(area_df[col].dtype)
    
    # Check zones
    if "zone" in area_df.columns:
        validation_results["zones"] = area_df["zone"].unique().tolist()
    
    return validation_results


def print_data_quality_report(validation_results: Dict[str, any]) -> None:
    """
    Print a data quality report
    
    Args:
        validation_results: Results from validate_data_quality function
    """
    print(f"\n{'='*60}")
    print("📋 Data Quality Report")
    print(f"{'='*60}")
    
    print(f"Total Rows: {validation_results['total_rows']}")
    print(f"Total Columns: {validation_results['total_columns']}")
    
    if validation_results["zones"]:
        print(f"Zones Found: {', '.join(validation_results['zones'])}")
    
    if validation_results["missing_values"]:
        print(f"\n⚠️ Missing Values:")
        for col, count in validation_results["missing_values"].items():
            percentage = (count / validation_results["total_rows"]) * 100
            print(f"  {col}: {count} ({percentage:.1f}%)")
    else:
        print(f"\n✅ No missing values found")
    
    print(f"{'='*60}")
