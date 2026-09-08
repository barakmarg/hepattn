# comet_env.sh -- put COMET_API_KEY into the environment without keeping the
# key in the repository. Meant to be *sourced*, not executed.
#
# Usage:
#     source /path/to/comet_env.sh                  # optional: warn, keep going
#     source /path/to/comet_env.sh --require || exit 1   # hard-fail if absent
#     source /path/to/comet_env.sh <key>            # use this key explicitly
#     source /path/to/comet_env.sh --require <key>
#
# Resolution order, first hit wins:
#   1. a key passed as an argument;
#   2. the key file -- $COMET_KEY_FILE, default $HOME/.comet.env, a shell
#      fragment containing COMET_API_KEY=... ;
#   3. COMET_API_KEY already exported in the environment.
#
# The file outranks an inherited environment variable on purpose: a stale
# `export COMET_API_KEY=` in a shell profile would otherwise silently keep a
# rotated key alive. Pass the key as an argument to override for one run.
#
# Returns 0 if the key ended up set, 1 otherwise. `--require` additionally
# prints an error; without it a warning is printed and the caller continues,
# which is what scripts that do not actually log to Comet want.
#
# Setting up a new machine or account:
#     printf 'COMET_API_KEY=%s\n' "<your key>" > ~/.comet.env && chmod 600 ~/.comet.env

_comet_require=0
_comet_arg_key=""
for _comet_arg in "$@"; do
    case "$_comet_arg" in
        --require)  _comet_require=1 ;;
        --optional) _comet_require=0 ;;
        -*)         echo "comet_env.sh: ignoring unknown option '$_comet_arg'" >&2 ;;
        *)          _comet_arg_key="$_comet_arg" ;;
    esac
done

: "${COMET_KEY_FILE:=$HOME/.comet.env}"

if [ -n "$_comet_arg_key" ]; then
    COMET_API_KEY="$_comet_arg_key"
elif [ -f "$COMET_KEY_FILE" ]; then
    # shellcheck disable=SC1090
    . "$COMET_KEY_FILE"
else
    :   # fall back to whatever the caller or batch system exported
fi

if [ -n "${COMET_API_KEY:-}" ]; then
    export COMET_API_KEY
    _comet_status=0
else
    _comet_status=1
    if [ "$_comet_require" = 1 ]; then
        echo "error: COMET_API_KEY is not set, and no key file at $COMET_KEY_FILE" >&2
        echo "       This run logs to Comet and will fail without it. Fix with:" >&2
        echo "         printf 'COMET_API_KEY=%s\\n' \"<your key>\" > $COMET_KEY_FILE && chmod 600 $COMET_KEY_FILE" >&2
        echo "       or pass it directly: source ${BASH_SOURCE[0]} <key>" >&2
    else
        echo "comet_env.sh: no COMET_API_KEY found (looked in $COMET_KEY_FILE);" >&2
        echo "              continuing, since this script does not log to Comet." >&2
    fi
fi

unset _comet_require _comet_arg_key _comet_arg

# `return` when sourced (the intended use), `exit` if someone runs this file.
if [ "${BASH_SOURCE[0]}" != "$0" ]; then
    return $_comet_status
else
    exit $_comet_status
fi
