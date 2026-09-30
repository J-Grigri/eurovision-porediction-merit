"""Rebuild everything: python src/run_all.py"""
import analysis
import build_data
import evidence
import features
import forecast
import models
import report
import semi_analysis
import diaspora
import geography
import politics

for step in (build_data.main, features.main, analysis.main, evidence.main, models.main, semi_analysis.main, diaspora.main, geography.main, forecast.main, politics.main, report.main):
    print(f"\n=== {step.__module__}.{step.__name__} ===")
    step()
