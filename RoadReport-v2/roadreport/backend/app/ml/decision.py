"""Conservative decision policy, separate from learned probabilities."""
def decide(visual_score, observations, contact, impact, predictions=None, experimental=False):
    visual_supported = visual_score >= 0.75 and observations >= 2
    if predictions and not experimental:
        visual_supported = visual_supported and predictions.get('pothole_exists', 0) >= 0.70
    if visual_score < 0.25:
        return 'dismissed', 'weak_visual', 'Visual evidence below baseline gate; shock alone cannot establish a pothole'
    if contact['state']=='no_contact_expected':
        return ('confirmed' if visual_supported else 'needs_review'),'visual_no_contact','No tyre contact expected; absence of shock is not negative evidence'
    if contact['state']=='contact_expected' and impact.get('ambiguous'):
        return 'needs_review','ambiguous_impact','Several candidates share this impact interval; no exclusive association'
    if visual_supported and contact['state']=='contact_expected' and impact['shock']:
        if impact['imu_available'] and predictions and not experimental and predictions.get('impact_associated',0)>=.8:
            return 'confirmed','impact_supported','Visual evidence, expected tyre contact and trained IMU association agree'
        return 'confirmed','motion_correlated','Visual baseline supports pothole; temporally correlated motion is supporting evidence, not verified contact'
    if visual_supported:
        return 'confirmed','visual_only','Visual baseline supports finding; impact evidence is absent or insufficient'
    return 'needs_review','uncertain','Evidence remains mixed; request human inspection'
