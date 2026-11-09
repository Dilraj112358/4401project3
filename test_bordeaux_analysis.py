"""Checks for the analysis's consequential label and independence rules."""
import unittest

import numpy as np
import pandas as pd

from bordeaux_analysis import bootstrap, flags, group_keys, mask_pattern, norm, pool


def reviews(rows):
    defaults = dict(title="Example Pinot Noir", designation="Estate", country="US",
                    winery="Example", province="California", region_1="Napa",
                    variety="Pinot Noir", color="red", description="Cherry and tannins.")
    return pd.DataFrame([{**defaults, **r} for r in rows])


class LabelAndHoldoutTests(unittest.TestCase):
    def test_floral_rose_is_not_a_color_correction(self):
        frame = reviews([dict(description="Rose petals, cherry and fine tannins.")])
        self.assertFalse(flags(frame).any(axis=1).iloc[0])

    def test_rose_and_sparkling_styles_are_excluded(self):
        frame = reviews([dict(title="Example Rosé Pinot Noir"),
                         dict(title="Example Brut Chardonnay", variety="Chardonnay", color="white"),
                         dict(description="This is a sparkling wine.")])
        self.assertTrue(flags(frame).any(axis=1).all())

    def test_sweet_flavor_does_not_alone_mean_dessert_wine(self):
        frame = reviews([dict(description="Sweet cherry flavors finish dry.")])
        self.assertFalse(flags(frame).any(axis=1).iloc[0])

    def test_idioms_and_producer_names_are_not_styles(self):
        frame = reviews([dict(description="At first blush, this has cherry and tannins.",
                              title="Château du Port 2009 Pinot Noir")])
        self.assertFalse(flags(frame).any(axis=1).iloc[0])

    def test_explicit_orange_wine_is_excluded_but_orange_flavor_is_not(self):
        frame = reviews([dict(description='This “orange” wine was macerated on skins.',
                              variety="Gewürztraminer", color="white"),
                         dict(description="Orange peel and apple flavors.",
                              variety="Gewürztraminer", color="white")])
        self.assertTrue(flags(frame).style_in_description.iloc[0])
        self.assertFalse(flags(frame).any(axis=1).iloc[1])

    def test_unknown_variety_is_excluded_without_relabeling(self):
        frame = reviews([dict(variety="Unknown grape")])
        self.assertTrue(flags(frame).outside_confident_varieties.iloc[0])
        self.assertEqual(frame.color.iloc[0], "red")

    def test_conflicting_duplicate_text_is_removed(self):
        frame = reviews([{}, dict(color="white", variety="Chardonnay")])
        self.assertTrue(pool(frame).empty)

    def test_normalization_catches_whitespace_duplicates(self):
        frame = reviews([dict(description=" CHERRY\n and  Tannins. "), {}])
        self.assertEqual(len(set(norm(frame.description))), 1)
        self.assertEqual(len(pool(frame)), 1)

    def test_wine_family_groups_vintages_and_handles_missing_designation(self):
        frame = reviews([dict(title="Example 2010 Pinot Noir", designation=None),
                         dict(title="Example 2011 Pinot Noir", designation=np.nan),
                         dict(winery="Different producer", designation=None)])
        groups = group_keys(frame)
        self.assertEqual(groups.iloc[0], groups.iloc[1])
        self.assertNotEqual(groups.iloc[0], groups.iloc[2])

    def test_confusion_bootstrap_respects_perfect_predictions(self):
        ci = bootstrap(["red", "red", "white", "white"], ["red", "red", "white", "white"], repeats=100)
        self.assertEqual(ci["accuracy"], dict(low=1.0, high=1.0))

    def test_mask_removes_grape_color_cues_and_preserves_fruit(self):
        text = mask_pattern().sub(" ", "White Chardonnay, apple, pear and citrus; red Cabernet.")
        self.assertIn("apple, pear and citrus", text)
        self.assertNotIn("Chardonnay", text)
        self.assertNotIn("Cabernet", text)
        self.assertNotIn("White", text)


if __name__ == "__main__":
    unittest.main()
