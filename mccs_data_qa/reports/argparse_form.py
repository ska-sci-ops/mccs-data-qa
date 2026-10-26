"""argparse_form: Describe an argparse parser as form fields, and turn field values back into argv."""

import argparse
import shlex
from dataclasses import dataclass

# Field kinds. switch: store_true; select: choices; multiselect: choices with nargs '+'/'*';
# number: int/float; list: nargs '+'/'*' without choices, as space-separated text; text: anything else


@dataclass
class Field:
    """One argparse argument as a form field."""

    dest: str
    flag: str | None  # Longest option string, None for a positional
    kind: str
    default: object
    required: bool
    help: str
    group: str
    nargs: str | int | None
    choices: list | None
    type: type | None


def _help(action: argparse.Action) -> str:
    """Help text with its %(default)s style placeholders filled."""
    try:
        return (action.help or '') % {**vars(action), 'default': action.default}
    except (KeyError, TypeError, ValueError):
        return action.help or ''


def _kind(action: argparse.Action) -> str:
    """Form field kind of an argparse action."""
    if isinstance(action, argparse._StoreTrueAction):
        return 'switch'
    if not isinstance(action, argparse._StoreAction):
        raise NotImplementedError(f'argparse action {type(action).__name__} for {action.dest!r}')
    multi = action.nargs in ('+', '*')
    if action.choices:
        return 'multiselect' if multi else 'select'
    if multi:
        return 'list'
    return 'number' if action.type in (int, float) else 'text'


def form_fields(parser: argparse.ArgumentParser) -> list[Field]:
    """Form fields for each argument of a parser, in argparse group order.

    Args:
        parser (argparse.ArgumentParser): Parser to describe.

    Returns:
        fields (list[Field]): One per argument, excluding -h. Group titles are in `Field.group`.

    Raises:
        NotImplementedError: For an argparse action other than store and store_true.
    """
    fields = []
    for group in parser._action_groups:
        for action in group._group_actions:
            if isinstance(action, argparse._HelpAction):
                continue
            positional = not action.option_strings
            fields.append(
                Field(
                    dest=action.dest,
                    flag=None if positional else max(action.option_strings, key=len),
                    kind=_kind(action),
                    default=action.default,
                    required=(action.nargs not in ('?', '*')) if positional else action.required,
                    help=_help(action),
                    group=group.title or '',
                    nargs=action.nargs,
                    choices=list(action.choices) if action.choices else None,
                    type=action.type,
                )
            )
    return fields


def default_values(fields: list[Field]) -> dict:
    """Initial form value of each field, by dest. 'list' fields are text, 'multiselect' fields lists."""
    values = {}
    for f in fields:
        if f.kind == 'switch':
            values[f.dest] = bool(f.default)
        elif f.kind == 'multiselect':
            values[f.dest] = list(f.default or [])
        elif f.kind == 'list':
            values[f.dest] = shlex.join(map(str, f.default or []))
        elif f.kind == 'number':
            values[f.dest] = f.default
        else:
            values[f.dest] = '' if f.default is None else f.default
    return values


def missing_required(fields: list[Field], values: dict) -> list[str]:
    """Dests of required fields that are empty."""
    return [f.dest for f in fields if f.required and values.get(f.dest) in (None, '', [])]


def _tokens(f: Field, v) -> list[str]:
    """Values of a nargs '+'/'*' field as argv tokens."""
    return shlex.split(v) if f.kind == 'list' else [str(x) for x in v]


def _number(f: Field, v) -> str:
    """A number field's value as an argv token."""
    return str(int(v)) if f.type is int else repr(float(v))


def build_argv(fields: list[Field], values: dict) -> list[str]:
    """Command-line arguments for form values: positionals first, then options that differ from their default.

    Args:
        fields (list[Field]): Output of `form_fields()`.
        values (dict): Value per dest, as from `default_values()`.

    Returns:
        argv (list[str]): Arguments for the script's own parser. Single-valued options are `--flag=value`
            so that values starting with '-' parse.
    """
    pos, opts = [], []
    for f in fields:
        v = values[f.dest]
        if f.flag is None:
            if f.nargs in ('+', '*'):
                pos += _tokens(f, v)
            elif v not in (None, ''):
                pos.append(str(v))
        elif f.kind == 'switch':
            if bool(v) != bool(f.default):
                opts.append(f.flag)
        elif f.kind in ('list', 'multiselect'):
            tokens = _tokens(f, v)
            if tokens != [str(x) for x in (f.default or [])] and (tokens or f.nargs == '*'):
                opts += [f.flag, *tokens]
        elif f.kind == 'number':
            if v is not None and (f.default is None or float(v) != float(f.default)):
                opts.append(f'{f.flag}={_number(f, v)}')
        elif v not in (None, '') and v != f.default:
            opts.append(f'{f.flag}={v}')
    return pos + opts
