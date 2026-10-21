"""
Run the alphaML 2.0 web application: python -m alphaml.web (or the alphaml-web command).

    alphaml-web [--auth] [--port 8000] [--home FOLDER]     start the server
    alphaml-web users list | add | reset-password | delete | claim [EMAIL]     manage accounts
"""
import argparse
import getpass
import json
import os
import shutil
import sys


def _owned_folders(home, user_id):
    """Data set and job folders that belong to a user (user_id None: folders that belong to nobody)."""
    for kind, record_name in (('datasets', 'dataset.json'), ('jobs', 'job.json')):
        for folder in sorted((home / kind).glob('*')):
            try:
                record = json.loads((folder / record_name).read_text(encoding='utf-8'))
            except (OSError, json.JSONDecodeError):
                continue
            if record.get('owner') == user_id and not record.get('public'):
                yield kind, folder, record_name, record


def _new_password():
    password = getpass.getpass('New password: ')
    if getpass.getpass('Repeat the password: ') != password:
        sys.exit('The passwords do not match.')
    return password


def users_command(args, home):
    from .auth import AuthError, UserStore

    store = UserStore(home / 'users.db')
    user = store.get(args.email) if args.email else None
    if args.action != 'list' and args.action != 'add' and user is None:
        sys.exit(f'No account for {args.email!r}.')
    try:
        if args.action == 'list':
            for u in store.list():
                owned = list(_owned_folders(home, u['id']))
                print(f"{u['email']:40s} created {u['created'][:10]}  "
                      f"{sum(k == 'datasets' for k, *_ in owned)} data sets, {sum(k == 'jobs' for k, *_ in owned)} jobs")
            unowned = list(_owned_folders(home, None))
            if unowned:
                print(f'{len(unowned)} data sets/jobs from before accounts were enabled belong to nobody '
                      f'(assign them with: users claim EMAIL)')
        elif args.action == 'add':
            if not args.email:
                sys.exit('Give the email address of the new account.')
            print(f"Created {store.create_user(args.email, _new_password())['email']}")
        elif args.action == 'reset-password':
            store.set_password(user['id'], _new_password())
            print(f"Password changed for {user['email']}; their sessions were ended.")
        elif args.action == 'delete':
            owned = list(_owned_folders(home, user['id']))
            if not args.yes and input(f"Delete {user['email']} and their {len(owned)} data sets and jobs? [y/N] ") \
                    .strip().lower() != 'y':
                sys.exit('Nothing deleted.')
            for _, folder, *_ in owned:
                shutil.rmtree(folder, ignore_errors=True)
            store.delete(user['id'])
            print(f"Deleted {user['email']} and {len(owned)} data sets and jobs.")
        elif args.action == 'claim':
            owned = list(_owned_folders(home, None))
            for _, folder, record_name, record in owned:
                record['owner'] = user['id']
                (folder / record_name).write_text(json.dumps(record, indent=2), encoding='utf-8')
            print(f"{len(owned)} data sets and jobs now belong to {user['email']}.")
    except AuthError as exc:
        sys.exit(str(exc))


def main(argv=None):
    parser = argparse.ArgumentParser(prog='alphaml-web', description='Run the alphaML 2.0 web application.')
    parser.add_argument('--host', default=os.environ.get('ALPHAML_HOST', '127.0.0.1'),
                        help='interface to listen on (default 127.0.0.1, this computer only)')
    parser.add_argument('--port', type=int, default=int(os.environ.get('ALPHAML_PORT', 8000)))
    parser.add_argument('--home', default=os.environ.get('ALPHAML_HOME'),
                        help='folder for data sets and job results (default: Documents/alphaML)')
    parser.add_argument('--max-jobs', type=int, default=int(os.environ.get('ALPHAML_MAX_JOBS', 1)),
                        help='number of jobs that may run at the same time (default 1)')
    parser.add_argument('--auth', action='store_true', default=None,
                        help='require accounts: each user sees only their own data (also ALPHAML_AUTH=1)')
    parser.add_argument('--retention-days', type=float, default=None,
                        help='delete data sets and results after this many days, 0 = keep '
                             '(default 14 with accounts, otherwise 0)')
    commands = parser.add_subparsers(dest='command')
    users = commands.add_parser('users', help='manage accounts (with --auth)')
    users.add_argument('action', choices=['list', 'add', 'reset-password', 'delete', 'claim'])
    users.add_argument('email', nargs='?')
    users.add_argument('--yes', action='store_true', help='delete without asking')
    args = parser.parse_args(argv)

    from .app import create_app, data_home

    home = data_home(args.home)
    if args.command == 'users':
        return users_command(args, home)

    import uvicorn

    app = create_app(home=home, max_jobs=args.max_jobs, auth=args.auth, retention_days=args.retention_days)
    print(f'alphaML 2.0 is running at http://{args.host}:{args.port}  (Ctrl+C to stop)')
    print(f'Data sets and results: {home}')
    print('Accounts: on, each user sees only their own data' if app.state.users else 'Accounts: off (single user)')
    if app.state.retention_days:
        print(f'Data sets and results are deleted {app.state.retention_days:g} days after they were created')
    uvicorn.run(app, host=args.host, port=args.port, log_level='warning')


if __name__ == '__main__':
    main()
