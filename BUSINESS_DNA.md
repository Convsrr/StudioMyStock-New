# Business DNA

This is what StudioMyStock is, who it's for, and the principles every product and engineering decision should pass through.

## What we do

We turn raw, phone-quality car photos into catalog-grade studio shots. Upload a snap on the forecourt, pick a studio background, get back a professional listing image in under a minute.

## Who it's for

**Primary: car dealerships** (independent and franchised) listing stock on their own site, AutoTrader, eBay Motors, Cars & Bids, Bring a Trailer, etc. They have hundreds of cars to photograph, no in-house studio, no patience for Photoshop.

**Secondary: marketplaces** that want to standardize listing imagery without making sellers do the work. They embed our pipeline.

**Tertiary: enthusiasts and shop owners** preparing builds for sale or social.

We are not for: stock photo agencies, ad creative shops, or anyone who needs a hero campaign image. They have budgets and studios. We're for the long tail.

## The problem we actually solve

A used-car listing with a bad photo loses ~30–50% of click-through. Dealers know this and still don't have time to drive every car to a studio. The current options are:

- **Hire a photographer** — too slow, too expensive at volume
- **DIY with phone + lightroom presets** — inconsistent, time-consuming
- **Existing AI tools** — change the car. Plate digits drift, wheel patterns shift, paint colour bleeds. For an automotive listing that's a non-starter; we sell the *exact* car.

Our wedge: **identity-preserving harmonization**. We get the studio look from a generative model, then paste the original car pixels back over the result. The buyer sees the same car they'll be paying for.

## What "good" looks like

A processed image is good when:

1. The car is **identifiable down to the plate digits** vs. the original
2. The **lighting feels like a real studio**, not a Photoshop comp
3. **Wheel contact** lands on the floor line — no floating cars
4. The **shadow direction** matches the studio's lighting
5. There are **no AI artifacts** in the visible car region

If any one of these breaks, the image fails. Our retention story depends on consistent passing on all five.

## Principles

These are non-negotiable. Use them when a tradeoff comes up.

### 1. Identity over aesthetics

If a generative pass beautifies the scene at the cost of changing the car, we revert the car. The dealer is selling a specific VIN, not a vibe.

### 2. The car must look like *this* car

Plates, badges, wheel patterns, dents, trim — all preserved. If a stage can't preserve them, it gets gated behind a flag and disabled by default.

### 3. Speed is a feature

A dealer with 200 cars to list this week won't wait 5 minutes per image. Sub-60-second target end-to-end on a phone-resolution input. Skip optional stages by default. Cache aggressively. Idempotency on identical inputs.

### 4. Quality > coverage

We'd rather process a 2008 Civic perfectly than half-process a fleet of supercars. Saying "we don't support this image" is fine. Shipping a broken result is not.

### 5. Boring infrastructure

FastAPI, Postgres-or-SQLite, Redis-or-inline, S3-or-disk. No exotic dependencies. The interesting work is in the pipeline, not the platform.

### 6. Determinism wherever we can get it

Same input + same params should give the same output. We cache, hash, and version every external model. When a provider releases a new model version we pin and compare before adopting.

### 7. Fail loud, recover quietly

External providers will fail. The pipeline retries with backoff, falls back to alternative providers (e.g. Picsart → Replicate for segmentation), and surfaces a typed error to the API if it can't recover. Stuck jobs are cleaned up by a cron script.

### 8. Respect the dealer's brand

Every output is theirs. Watermarks, custom backgrounds, brand presets, plate blur — all opt-in, all under their control.

## How we make money

The unit economics work because each image consumes a small, predictable slice of compute:

- One Picsart segmentation call (cents)
- One Qwen Image Edit call on Replicate (cents)
- Optional upscale (cents)
- Storage (negligible at car-listing scale)

Pricing model (early thinking, not committed):

- **Pay-as-you-go** per image for hobbyists
- **Volume tiers** for dealerships (monthly, with overage)
- **API for marketplaces** with revenue share or wholesale pricing

Gross margin target: 70%+. We don't chase markets where we can't hit that.

## What success looks like (12 months)

- 200+ active dealerships across UK and US
- 1M+ images processed
- < 1% support-ticket rate
- < 0.1% identity-drift complaints (the car looking different)
- One marketplace integration in production

## What we explicitly say no to

- Editing the car (changing colour, swapping wheels, virtual tuning) — that's a different product, possibly a future one, but not our wedge
- Outdoor / lifestyle scenes — generative outdoor shots have ground-truth problems we can't solve cheaply
- Video — orders of magnitude more compute, completely different UX
- "AI photographer" generic positioning — every Y Combinator batch has three of those. We're vertical: cars only.

## Brand voice

Practical, confident, dealer-first. We say "your stock", "your forecourt", "your listings". Not "AI-powered cutting-edge synthesis". The dealer doesn't care how it works, only that it works.

When something fails we say so plainly and offer the next step. No hand-waving.

## How to use this doc

Before merging a feature, ask:

1. Does this make our images more identity-preserving?
2. Does this make us faster at scale?
3. Does this make a dealer's life easier?

If the answer is no to all three, it probably doesn't ship.
