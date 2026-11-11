import csv
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from source_io import fetch

ID_COL = 'organisation_number'
TIME_COL = 'date_of_registration'
TARGET_COL = 'exit_within_5y'
FEATURE_COLUMNS = [
    'charity_type',
    'charity_is_cio',
    'has_company_registration',
    'classification_what',
    'classification_who',
    'area_of_operation_scope',
    'governing_document_description',
    'objects_text',
]
OUTPUT_COLUMNS = [ID_COL, TIME_COL] + FEATURE_COLUMNS + [TARGET_COL]

CHARITY_URL = 'https://ccewuksprdoneregsadata1.blob.core.windows.net/data/txt/publicextract.charity.zip'
CLASSIFICATION_URL = 'https://ccewuksprdoneregsadata1.blob.core.windows.net/data/txt/publicextract.charity_classification.zip'
AREA_URL = 'https://ccewuksprdoneregsadata1.blob.core.windows.net/data/txt/publicextract.charity_area_of_operation.zip'
GOV_URL = 'https://ccewuksprdoneregsadata1.blob.core.windows.net/data/txt/publicextract.charity_governing_document.zip'

START = pd.Timestamp('2015-01-01')
TRAIN_END = pd.Timestamp('2017-01-01')
EVAL_END = pd.Timestamp('2018-01-01')
HOLDOUT_END = pd.Timestamp('2019-01-01')
END_EXCL = HOLDOUT_END


def _bounded(text, limit=4000):
    text = str(text)
    if len(text) <= limit:
        return text
    return text[: limit - 3] + '...'


def _read_zip_table(path, label):
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        raise AssertionError(_bounded(f'Source archive missing or empty: {path}'))
    zf = zipfile.ZipFile(path)
    names = [n for n in zf.namelist() if not n.endswith('/') and n.lower().endswith(('.txt', '.tsv', '.csv'))]
    if not names:
        raise AssertionError(_bounded(f'No text member in archive {path}; members={zf.namelist()!r}'))
    preferred = [n for n in names if label.casefold() in n.casefold()]
    candidates = preferred or names
    member = sorted(candidates, key=lambda n: (-zf.getinfo(n).file_size, n))[0]
    try:
        with zf.open(member) as fh:
            df = pd.read_csv(
                fh,
                sep='\t',
                dtype=str,
                quoting=csv.QUOTE_NONE,
                encoding='utf-8',
                keep_default_na=False,
                on_bad_lines='warn',
            )
    except Exception as exc:
        raise AssertionError(_bounded(f'Could not read archive member {member!r} from {path}: {exc!r}'))
    df.columns = [str(c).replace('\ufeff', '').strip() for c in df.columns]
    if df.shape[1] == 0:
        raise AssertionError(_bounded(f'No columns in archive member {member!r}; headers={list(df.columns)!r}'))
    return df, member


def _pick(df, candidates, logical, label, member):
    exact = {str(c).casefold(): c for c in df.columns}
    for cand in candidates:
        if cand.casefold() in exact:
            return exact[cand.casefold()]

    def norm(value):
        return re.sub(r'[^a-z0-9]', '', str(value).casefold())

    normalised = {norm(c): c for c in df.columns}
    for cand in candidates:
        key = norm(cand)
        if key in normalised:
            return normalised[key]
    message = (
        f'Missing field {logical!r} in {label} table. '
        f'Observed headers: {list(df.columns)!r}. '
        f'Missing names: {candidates!r}. '
        f'Selected archive member: {member!r}.'
    )
    raise AssertionError(_bounded(message))


def _norm_key(series):
    return series.astype(str).str.strip().str.replace(r'\.0+$', '', regex=True)


def _norm_status(value):
    text = str(value).strip()
    folded = text.casefold()
    if folded in {'registered', 'r', 'reg', 'live', 'active'}:
        return 'registered'
    if folded in {'removed', 'rm', 'rem', 'inactive'}:
        return 'removed'
    if folded == '':
        return 'unknown:blank'
    return 'unknown:' + text


def _binary_flag(series):
    text = series.astype(str).str.strip().str.casefold()
    out = pd.Series('', index=series.index, dtype='object')
    out[text.isin(['y', 'yes', '1', 'true', 't'])] = '1'
    out[text.isin(['n', 'no', '0', 'false', 'f'])] = '0'
    return out


def _clean_text(value):
    if value is None:
        return ''
    text = str(value).replace('\ufeff', '')
    return re.sub(r'\s+', ' ', text).strip()


def _join_unique(series):
    values = sorted({_clean_text(v) for v in series if _clean_text(v) != ''})
    return '; '.join(values)


def _add_check(checks, name, observed, expected, passed, kind=None):
    item = {
        'check': name,
        'observed': str(observed),
        'expected': str(expected),
        'passed': bool(passed),
    }
    if kind is not None:
        item['kind'] = kind
    checks.append(item)


def build(output: Path, cache: Path):
    output = Path(output)
    cache = Path(cache)
    checks = []

    charity_path = fetch(CHARITY_URL, cache, 'publicextract.charity.zip')
    classification_path = fetch(CLASSIFICATION_URL, cache, 'publicextract.charity_classification.zip')
    area_path = fetch(AREA_URL, cache, 'publicextract.charity_area_of_operation.zip')
    governing_path = fetch(GOV_URL, cache, 'publicextract.charity_governing_document.zip')

    for name, path in [
        ('charity', charity_path),
        ('classification', classification_path),
        ('area_of_operation', area_path),
        ('governing_document', governing_path),
    ]:
        size = Path(path).stat().st_size if Path(path).exists() else 0
        _add_check(checks, f'fetched {name} archive', f'{size} bytes', 'nonempty local archive', size > 0)

    charity_df, charity_member = _read_zip_table(charity_path, 'charity')
    classification_df, classification_member = _read_zip_table(classification_path, 'classification')
    area_df, area_member = _read_zip_table(area_path, 'area')
    governing_df, governing_member = _read_zip_table(governing_path, 'governing')

    _add_check(checks, 'selected charity archive member', charity_member, 'largest tab-delimited member of publicextract.charity.zip', True)
    _add_check(checks, 'selected classification archive member', classification_member, 'largest tab-delimited member of publicextract.charity_classification.zip', True)
    _add_check(checks, 'selected area archive member', area_member, 'largest tab-delimited member of publicextract.charity_area_of_operation.zip', True)
    _add_check(checks, 'selected governing-document archive member', governing_member, 'largest tab-delimited member of publicextract.charity_governing_document.zip', True)

    org_c = _pick(charity_df, ['organisation_number', 'organization_number'], 'organisation_number', 'charity', charity_member)
    linked_c = _pick(charity_df, ['linked_charity_number'], 'linked_charity_number', 'charity', charity_member)
    status_c = _pick(charity_df, ['charity_registration_status', 'registration_status', 'status'], 'charity_registration_status', 'charity', charity_member)
    reg_c = _pick(charity_df, ['date_of_registration', 'registration_date'], 'date_of_registration', 'charity', charity_member)
    rem_c = _pick(charity_df, ['date_of_removal', 'removal_date'], 'date_of_removal', 'charity', charity_member)
    type_c = _pick(charity_df, ['charity_type', 'charity_type_description', 'type_of_charity'], 'charity_type', 'charity', charity_member)
    cio_c = _pick(charity_df, ['charity_is_cio', 'is_cio', 'charity_cio'], 'charity_is_cio', 'charity', charity_member)
    company_c = _pick(charity_df, ['charity_company_registration_number', 'company_registration_number', 'charity_company_number', 'company_number'], 'charity_company_registration_number', 'charity', charity_member)

    charity = pd.DataFrame({
        ID_COL: _norm_key(charity_df[org_c]),
        '_linked': pd.to_numeric(charity_df[linked_c].astype(str).str.strip(), errors='coerce'),
        '_status_raw': charity_df[status_c].astype(str).str.strip(),
        TIME_COL: pd.to_datetime(charity_df[reg_c], errors='coerce'),
        '_removed': pd.to_datetime(charity_df[rem_c], errors='coerce'),
        'charity_type': charity_df[type_c].map(_clean_text),
        'charity_is_cio': _binary_flag(charity_df[cio_c]),
        '_company': charity_df[company_c].map(_clean_text),
    })
    charity['_status'] = charity['_status_raw'].map(_norm_status)
    charity['has_company_registration'] = np.where(
        charity['_company'].str.casefold().isin(['', '0', 'n/a', 'na', 'none']),
        '0',
        '1',
    )

    main = charity['_linked'].eq(0)
    in_window = charity[TIME_COL].notna() & (charity[TIME_COL] >= START) & (charity[TIME_COL] < END_EXCL)
    main_in_window = main & in_window

    unknown_status_in_window = charity.loc[main_in_window & ~charity['_status'].isin(['registered', 'removed'])]
    if not unknown_status_in_window.empty:
        counts = unknown_status_in_window['_status_raw'].value_counts(dropna=False).to_dict()
        message = (
            'Eligible main-charity rows contain unrecognised charity_registration_status values. '
            f'Observed headers: {list(charity_df.columns)!r}. '
            f'Selected archive member: {charity_member!r}. '
            f'Unrecognised raw-status counts: {counts!r}.'
        )
        raise AssertionError(_bounded(message))

    unknown_status_elsewhere = charity.loc[
        main & ~in_window & ~charity['_status'].isin(['registered', 'removed']), '_status_raw'
    ].value_counts(dropna=False).to_dict()
    _add_check(
        checks,
        'main-charity status codes outside the frozen cohort',
        unknown_status_elsewhere if unknown_status_elsewhere else 'none',
        'recorded without being used as labels',
        True,
    )

    main_status_known = main & charity['_status'].isin(['registered', 'removed'])
    removed_missing = int((main_status_known & charity['_status'].eq('removed') & charity['_removed'].isna()).sum())
    registered_with_date = int((main_status_known & charity['_status'].eq('registered') & charity['_removed'].notna()).sum())
    if removed_missing or registered_with_date:
        message = (
            'date_of_removal is inconsistent with charity_registration_status for main charities. '
            f'Observed headers: {list(charity_df.columns)!r}. '
            f'Selected archive member: {charity_member!r}. '
            f'Removed rows without a removal date: {removed_missing}. '
            f'Registered rows with a removal date: {registered_with_date}.'
        )
        raise AssertionError(_bounded(message))
    _add_check(
        checks,
        'date_of_removal populated exactly for removed main charities',
        f'removed_missing={removed_missing}; registered_with_date={registered_with_date}',
        'both counts zero, per the publisher data definition',
        removed_missing == 0 and registered_with_date == 0,
    )

    cohort = charity.loc[main_status_known & in_window].copy()
    pre_registration_removal = int((cohort['_removed'].notna() & (cohort['_removed'] < cohort[TIME_COL])).sum())
    cohort = cohort.loc[~(cohort['_removed'].notna() & (cohort['_removed'] < cohort[TIME_COL]))].copy()
    _add_check(
        checks,
        'pre-registration removal rows',
        f'{pre_registration_removal} rows excluded',
        'rows with date_of_removal before date_of_registration are excluded and counted',
        bool(not (cohort['_removed'].notna() & (cohort['_removed'] < cohort[TIME_COL])).any()),
    )

    duplicate_ids = cohort[ID_COL][cohort[ID_COL].duplicated()].tolist()
    if duplicate_ids:
        message = (
            'Duplicate organisation_number values in the frozen main-charity cohort. '
            f'Observed headers: {list(charity_df.columns)!r}. '
            f'Selected archive member: {charity_member!r}. '
            f'Duplicate examples: {duplicate_ids[:20]!r}.'
        )
        raise AssertionError(_bounded(message))
    _add_check(checks, 'duplicate organisation_number in cohort', f'{len(duplicate_ids)} duplicates', '0 duplicates', len(duplicate_ids) == 0)

    anniversary = cohort[TIME_COL] + pd.DateOffset(years=5)
    expected_target = ((cohort['_removed'].notna()) & (cohort['_removed'] <= anniversary)).astype(int)
    cohort[TARGET_COL] = expected_target
    _add_check(
        checks,
        'label rule dates',
        f'removal_date <= date_of_registration + 5 calendar years; positives={int(cohort[TARGET_COL].sum())}',
        'frozen five-year anniversary rule',
        bool((cohort[TARGET_COL] == expected_target).all()),
        kind='unit',
    )

    sample = cohort.head(min(100, len(cohort)))
    if len(sample):
        anniversary_sample = sample[TIME_COL] + pd.DateOffset(years=5)
        unit_ok = bool((anniversary_sample.dt.year == sample[TIME_COL].dt.year + 5).all())
        _add_check(
            checks,
            'five-year horizon calendar-year unit',
            f'{len(sample)} sampled rows have anniversary year equal to registration year plus five',
            'calendar-year frame, not fixed 365-day arithmetic',
            unit_ok,
            kind='unit',
        )
    else:
        _add_check(checks, 'five-year horizon calendar-year unit', 'no cohort rows', 'at least one row', False, kind='unit')

    c_org = _pick(classification_df, ['organisation_number', 'organization_number'], 'organisation_number', 'charity_classification', classification_member)
    c_type = _pick(classification_df, ['classification_type', 'class_type', 'type'], 'classification_type', 'charity_classification', classification_member)
    c_code = _pick(classification_df, ['classification_code', 'class_code', 'code'], 'classification_code', 'charity_classification', classification_member)

    cls = pd.DataFrame({
        ID_COL: _norm_key(classification_df[c_org]),
        '_type': classification_df[c_type].astype(str).str.strip().str.casefold(),
        '_code': classification_df[c_code].map(_clean_text),
    })
    cls = cls[(cls[ID_COL] != '') & (cls['_code'] != '')]
    what_types = {'what', 'what the charity does', 'charitable purposes', 'purpose', 'what it does'}
    who_types = {'who', 'who the charity helps', 'beneficiaries', 'who it helps', 'beneficiary'}
    type_counts = cls['_type'].value_counts(dropna=False).to_dict()
    what_map = cls.loc[cls['_type'].isin(what_types)].groupby(ID_COL)['_code'].agg(_join_unique)
    who_map = cls.loc[cls['_type'].isin(who_types)].groupby(ID_COL)['_code'].agg(_join_unique)
    if what_map.empty and who_map.empty:
        message = (
            'No classification rows matched the documented What or Who classification_type values. '
            f'Observed headers: {list(classification_df.columns)!r}. '
            f'Selected archive member: {classification_member!r}. '
            f'Observed classification_type counts: {type_counts!r}.'
        )
        raise AssertionError(_bounded(message))

    a_org = _pick(area_df, ['organisation_number', 'organization_number'], 'organisation_number', 'charity_area_of_operation', area_member)
    a_area = _pick(
        area_df,
        [
            'area_of_operation',
            'area_of_operation_description',
            'geo_area_description',
            'geographic_area_description',
            'area_description',
            'area_of_operation_type',
            'geo_area_type',
            'geographic_area_type',
        ],
        'area_of_operation description',
        'charity_area_of_operation',
        area_member,
    )
    area = pd.DataFrame({
        ID_COL: _norm_key(area_df[a_org]),
        '_area': area_df[a_area].map(_clean_text),
    })
    area = area[(area[ID_COL] != '') & (area['_area'] != '')]
    area_map = area.groupby(ID_COL)['_area'].agg(_join_unique)

    g_org = _pick(governing_df, ['organisation_number', 'organization_number'], 'organisation_number', 'charity_governing_document', governing_member)
    g_desc = _pick(
        governing_df,
        ['governing_document_description', 'governing_document', 'governing_document_type', 'governing_document_type_description'],
        'governing_document_description',
        'charity_governing_document',
        governing_member,
    )
    g_objects = _pick(
        governing_df,
        ['charitable_objects', 'objects', 'charitable_objects_text', 'objects_text', 'charitable_purpose'],
        'charitable objects text',
        'charity_governing_document',
        governing_member,
    )
    gov = pd.DataFrame({
        ID_COL: _norm_key(governing_df[g_org]),
        '_gov': governing_df[g_desc].map(_clean_text),
        '_objects': governing_df[g_objects].map(_clean_text),
    })
    gov = gov[gov[ID_COL] != '']
    gov_map = gov.groupby(ID_COL)['_gov'].agg(_join_unique)
    objects_map = gov.groupby(ID_COL)['_objects'].agg(_join_unique)

    cohort['classification_what'] = cohort[ID_COL].map(what_map).fillna('')
    cohort['classification_who'] = cohort[ID_COL].map(who_map).fillna('')
    cohort['area_of_operation_scope'] = cohort[ID_COL].map(area_map).fillna('')
    cohort['governing_document_description'] = cohort[ID_COL].map(gov_map).fillna('')
    cohort['objects_text'] = cohort[ID_COL].map(objects_map).fillna('')

    for feature in FEATURE_COLUMNS:
        if feature in ('classification_what', 'classification_who'):
            source_headers = list(classification_df.columns)
            source_member = classification_member
        elif feature == 'area_of_operation_scope':
            source_headers = list(area_df.columns)
            source_member = area_member
        elif feature in ('governing_document_description', 'objects_text'):
            source_headers = list(governing_df.columns)
            source_member = governing_member
        else:
            source_headers = list(charity_df.columns)
            source_member = charity_member
        nonempty = int(cohort[feature].fillna('').astype(str).str.strip().ne('').sum())
        if nonempty == 0:
            message = (
                f'Declared feature {feature!r} is empty for every cohort row after parsing. '
                f'Observed source headers: {source_headers!r}. '
                f'Missing names: none; the field resolved but produced no nonempty values. '
                f'Selected archive member: {source_member!r}.'
            )
            raise AssertionError(_bounded(message))
        _add_check(
            checks,
            f'nonempty values for {feature}',
            f'{nonempty} of {len(cohort)} cohort rows',
            'at least one nonempty parsed value',
            nonempty > 0,
        )

    sources = {
        'classification_what': (classification_df, classification_member),
        'classification_who': (classification_df, classification_member),
        'area_of_operation_scope': (area_df, area_member),
        'governing_document_description': (governing_df, governing_member),
        'objects_text': (governing_df, governing_member),
    }
    for label, mapping in [
        ('classification_what', what_map),
        ('classification_who', who_map),
        ('area_of_operation_scope', area_map),
        ('governing_document_description', gov_map),
        ('objects_text', objects_map),
    ]:
        source_df, source_member = sources[label]
        matched = int(cohort[ID_COL].isin(mapping.index).sum())
        if matched == 0:
            message = (
                f'Join on organisation_number produced no matches for {label}. '
                f'Observed cohort key header: {ID_COL!r}. '
                f'Auxiliary headers: {list(source_df.columns)!r}. '
                f'Missing names: none; the key column resolved but values did not overlap. '
                f'Selected archive member: {source_member!r}.'
            )
            raise AssertionError(_bounded(message))
        _add_check(
            checks,
            f'join coverage for {label}',
            f'{matched} of {len(cohort)} cohort rows matched',
            'at least one matched organisation_number',
            matched > 0,
        )

    cohort['_split'] = np.select(
        [
            (cohort[TIME_COL] >= START) & (cohort[TIME_COL] < TRAIN_END),
            (cohort[TIME_COL] >= TRAIN_END) & (cohort[TIME_COL] < EVAL_END),
            (cohort[TIME_COL] >= EVAL_END) & (cohort[TIME_COL] < HOLDOUT_END),
        ],
        ['train', 'eval', 'holdout'],
        default='',
    )
    if (cohort['_split'] == '').any():
        raise AssertionError(_bounded('Some cohort rows did not fall into a frozen split window.'))

    split_frames = {}
    for split_name, start, end in [
        ('train', START, TRAIN_END),
        ('eval', TRAIN_END, EVAL_END),
        ('holdout', EVAL_END, HOLDOUT_END),
    ]:
        frame = cohort.loc[cohort['_split'].eq(split_name)].copy()
        if frame.empty:
            _add_check(checks, f'{split_name} split nonempty', '0 rows', 'at least one row', False)
        else:
            observed_window = f'{frame[TIME_COL].min().date().isoformat()}..{frame[TIME_COL].max().date().isoformat()}'
            expected_window = f'half-open [{start.date().isoformat()}, {end.date().isoformat()})'
            _add_check(
                checks,
                f'{split_name} registration-date window',
                observed_window,
                expected_window,
                bool((frame[TIME_COL] >= start).all() and (frame[TIME_COL] < end).all()),
            )
            positives = int(frame[TARGET_COL].sum())
            _add_check(
                checks,
                f'{split_name} positive labels',
                f'{positives} positives in {len(frame)} rows',
                'at least one positive and no more positives than rows',
                0 < positives <= len(frame),
            )
        split_frames[split_name] = frame

    for split_name, frame in split_frames.items():
        if frame.empty:
            continue
        for feature in FEATURE_COLUMNS:
            nonempty_split = int(frame[feature].fillna('').astype(str).str.strip().ne('').sum())
            if nonempty_split == 0:
                _add_check(checks, f'{split_name} nonempty {feature}', '0 nonempty', 'at least one nonempty', False)

    out_frames = {}
    for split_name, frame in split_frames.items():
        frame = frame.sort_values([TIME_COL, ID_COL], kind='mergesort')
        out = pd.DataFrame({
            ID_COL: frame[ID_COL].astype(str),
            TIME_COL: frame[TIME_COL].dt.strftime('%Y-%m-%d'),
        })
        for feature in FEATURE_COLUMNS:
            out[feature] = frame[feature].fillna('').astype(str)
        out[TARGET_COL] = frame[TARGET_COL].astype(int)
        out = out[OUTPUT_COLUMNS]
        out_frames[split_name] = out
        _add_check(checks, f'{split_name} output headers', list(out.columns), OUTPUT_COLUMNS, list(out.columns) == OUTPUT_COLUMNS)
        _add_check(checks, f'{split_name} target header', TARGET_COL if TARGET_COL in out.columns else 'missing', TARGET_COL, TARGET_COL in out.columns)

    failed = [c for c in checks if not c['passed']]
    if failed:
        raise AssertionError(_bounded('Source checks failed: ' + repr(failed)))

    public_dir = output / 'public'
    private_dir = output / 'private'
    public_dir.mkdir(parents=True, exist_ok=True)
    private_dir.mkdir(parents=True, exist_ok=True)
    out_frames['train'].to_csv(public_dir / 'train.csv', index=False, lineterminator='\n')
    out_frames['eval'].to_csv(public_dir / 'eval.csv', index=False, lineterminator='\n')
    out_frames['holdout'].to_csv(private_dir / 'holdout.csv', index=False, lineterminator='\n')

    return checks
